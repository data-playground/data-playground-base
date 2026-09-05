# airflow/agents/recipe_agents.py
"""
Recipe pipeline agent functions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL ROUTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent                  Provider     Model
─────────────────────  ───────────  ──────────────────────────
Recipe Extractor       Gemini       gemini-2.5-flash
Ingredient Normalizer  Gemma        gemma-4-31b-it
Recipe Discoverer      Gemini       gemini-2.5-flash

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE NOTE — TWO AGENTS, ONE CALL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

By design, `agent_normalize_ingredients` and `agent_categorize_ingredients`
are defined as separate functions. Currently they are wired together in
a single Gemma call (agent_normalize_ingredients calls both internally)
to conserve API quota.

To split them into two separate calls later:
  1. In the normalization prompt, remove the "category" field from the
     response schema.
  2. After normalization, call agent_categorize_ingredients() separately
     with just the canonical names of ingredients that are NEW to the DB.
  3. Update _run_normalization_pipeline() in routers/recipes.py accordingly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TODO: Playwright headless browser
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

URL extraction currently uses requests + Schema.org JSON-LD parsing
with a Gemini fallback for non-structured pages.

For JS-heavy sites (NYT Cooking, Bon Appétit, Serious Eats), the
requests approach silently returns a partial/empty HTML page.

Integration point: _fetch_url_content() in routers/recipe_extract.py.
When adding Playwright:
  1. pip install playwright && playwright install chromium
  2. In _fetch_url_content(), after the requests attempt, check if the
     returned HTML contains recognizable recipe content. If not, fall
     through to a Playwright async fetch.
  3. The rest of the pipeline (Schema.org parse → Gemini fallback)
     remains unchanged — Playwright just provides better raw HTML.
"""

import json
import logging
import re

from services.ai import MODEL_FLASH, call_gemini_json, call_gemma_json, call_gemini_vision_json

log = logging.getLogger(__name__)


def _safe_json(raw: str) -> any:
    """
    Strips markdown fences and parses JSON.
    Both Gemini and Gemma occasionally wrap JSON in ```json ... ``` blocks
    despite being told not to — this handles it defensively.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    return json.loads(cleaned)


# ── ALLOWED VALUES (mirrors models.py enums) ──────────────────────────────────

_ALLOWED_UNITS = {
    "cup", "tbsp", "tsp", "ml", "l", "g", "kg", "oz", "lb",
    "piece", "clove", "bunch", "slice", "can", "package",
    "to_taste", "as_needed", "pinch", "handful",
}

_ALLOWED_CATEGORIES = {
    "produce", "protein", "dairy", "grain", "pantry",
    "spice", "condiment", "beverage", "frozen", "other",
}

_ALLOWED_MEAL_TYPES = {
    "breakfast", "lunch", "dinner", "snack",
    "dessert", "side", "drink", "other",
}

_ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}

_ALLOWED_SOURCE_TYPES = {"manual", "url", "pdf", "image", "ai_generated"}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — INGREDIENT NORMALIZER
# Provider: Gemma (gemma-3-27b-it)
# Frequency: Once per recipe import
#
# Two logical agents in one call (see module docstring for split instructions):
#   agent_normalize_ingredients — strips prep notes, resolves canonical names
#   agent_categorize_ingredients — assigns ingredient category
#
# Both are defined as wrapper functions below so callers can reference them
# by name even though they share a single API call today.
# ─────────────────────────────────────────────────────────────────────────────

_NORMALIZATION_PROMPT_TEMPLATE = """You are an ingredient normalizer for a recipe database. Process this raw ingredient list and return a JSON array.

For each ingredient item, extract:
- canonical_name: The simple, singular ingredient name with NO preparation method and NO quantity. Examples: "garlic" (not "minced garlic cloves"), "chicken breast" (not "boneless skinless chicken breast, cubed"), "flour" (not "all-purpose flour, sifted").
- quantity: The numeric quantity as a decimal number, or null if not specified or if the ingredient is "to taste"/"as needed".
- unit: One of these exact strings only: cup, tbsp, tsp, ml, l, g, kg, oz, lb, piece, clove, bunch, slice, can, package, to_taste, as_needed, pinch, handful. Use null if no unit applies.
- preparation_note: The preparation method ONLY — "finely diced", "at room temperature", "roughly chopped", "boneless skinless", etc. null if none.
- is_optional: true if the ingredient is marked optional or "if desired", otherwise false.
- category: One of: produce, protein, dairy, grain, pantry, spice, condiment, beverage, frozen, other.

Rules:
- Strip ALL quantities and units from canonical_name — those go in quantity/unit fields.
- Strip ALL preparation methods from canonical_name — those go in preparation_note.
- canonical_name should be the simplest recognizable form of the ingredient.
- For "salt and pepper to taste" → two separate items: {{name:"salt", quantity:null, unit:"to_taste"}} and {{name:"black pepper", quantity:null, unit:"to_taste"}}.
- For "1 can (14 oz) diced tomatoes" → {{name:"diced tomatoes", quantity:1, unit:"can", preparation_note:null}}.

Respond with ONLY a JSON array. No markdown, no explanation.

Raw ingredient list:
{raw_ingredients}"""


def agent_normalize_ingredients(raw_ingredient_lines: list[str]) -> list[dict]:
    """
    Normalizes a raw ingredient list into structured records.
    """
    if not raw_ingredient_lines:
        return []

    joined = "\n".join(f"- {line}" for line in raw_ingredient_lines)
    prompt = _NORMALIZATION_PROMPT_TEMPLATE.format(raw_ingredients=joined)

    try:
        raw = call_gemma_json(prompt)
        result = _safe_json(raw)
    except Exception as exc:
        log.error("Ingredient normalization failed: %s", exc)
        return [
            {
                "canonical_name": line.strip()[:150],
                "quantity": None,
                "unit": None,
                "preparation_note": None,
                "is_optional": False,
                "category": "other",
            }
            for line in raw_ingredient_lines
        ]

    normalized = []
    for i, item in enumerate(result):
        if not isinstance(item, dict):
            continue
        unit = item.get("unit")
        if unit and unit not in _ALLOWED_UNITS:
            unit = None
        category = item.get("category", "other")
        if category not in _ALLOWED_CATEGORIES:
            category = "other"
        qty = item.get("quantity")
        try:
            qty = float(qty) if qty is not None else None
        except (ValueError, TypeError):
            qty = None

        normalized.append({
            "canonical_name": str(item.get("canonical_name", "unknown ingredient"))[:150].strip(),
            "quantity":        qty,
            "unit":            unit,
            "preparation_note": str(item.get("preparation_note", "") or "")[:150] or None,
            "is_optional":     bool(item.get("is_optional", False)),
            "category":        category,
        })

    log.info("Normalized %d ingredients from %d raw lines", len(normalized), len(raw_ingredient_lines))
    return normalized


def agent_categorize_ingredients(canonical_names: list[str]) -> dict[str, str]:
    """
    Assigns an ingredient category to each canonical ingredient name.
    """
    raise NotImplementedError(
        "agent_categorize_ingredients is not yet wired as a standalone call. "
        "Categorization is currently handled inside agent_normalize_ingredients. "
        "See the module docstring for split instructions."
    )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — RECIPE EXTRACTOR
# Provider: Gemini 2.5 Flash
# Frequency: Once per recipe import from URL, PDF, or image
#
# Two extraction paths:
#   Path A (Schema.org): parse JSON-LD directly — no AI needed.
#   Path B (Gemini fallback): called when Schema.org is absent.
#
# This agent handles Path B only. Path A is handled in recipe_extract.py
# by _parse_schema_org().
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title":               {"type": "STRING"},
        "cuisine":             {"type": "STRING"},
        "meal_type":           {"type": "STRING"},
        "prep_time_minutes":   {"type": "INTEGER"},
        "cook_time_minutes":   {"type": "INTEGER"},
        "total_time_minutes":  {"type": "INTEGER"},
        "servings":            {"type": "INTEGER"},
        "difficulty":          {"type": "STRING"},
        "instructions":        {"type": "STRING"},
        "notes":               {"type": "STRING"},
        "image_url":           {"type": "STRING"},
        "raw_ingredient_lines": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["title", "raw_ingredient_lines"],
}

_EXTRACTION_SYSTEM = """You are a recipe extractor. Extract structured recipe data from the provided text content.

Rules:
- title: The recipe name, clean and concise.
- cuisine: e.g. "Italian", "Mexican", "Japanese", "American", etc. null if unclear.
- meal_type: One of: breakfast, lunch, dinner, snack, dessert, side, drink, other. null if unclear.
- prep_time_minutes: Integer minutes, null if not stated.
- cook_time_minutes: Integer minutes, null if not stated.
- total_time_minutes: Integer minutes, null if not stated.
- servings: Integer number of servings, null if not stated.
- difficulty: One of: easy, medium, hard. Infer if not stated — a 4-ingredient 15-minute recipe is easy, a multi-day braise is hard.
- instructions: The full cooking instructions as a numbered Markdown list. Preserve all steps. Format: "1. Step one\\n2. Step two\\n..."
- notes: Any tips, variations, storage instructions, or chef notes. null if none.
- image_url: The URL of the main recipe image if you can identify one from the content. null if not found.
- raw_ingredient_lines: An array of ingredient strings exactly as written — do NOT parse or clean them. Include quantities, units, and preparation notes verbatim. e.g. ["2 cloves garlic, minced", "1 cup all-purpose flour", "salt and pepper to taste"]

If the content does not appear to contain a recipe, return an object with title="Unknown Recipe" and raw_ingredient_lines=[].
"""


def agent_extract_recipe(raw_content: str, source_hint: str = "") -> dict:
    """
    Extracts a structured recipe from unstructured text content.
    """
    hint_text = f" ({source_hint})" if source_hint else ""
    prompt = (
        f"Extract the recipe from this content{hint_text}:\n\n"
        f"{raw_content[:12000]}"
    )

    try:
        raw = call_gemini_json(prompt, schema=_EXTRACTION_SCHEMA, system=_EXTRACTION_SYSTEM)
        result = _safe_json(raw)
    except Exception as exc:
        log.error("Recipe extraction failed: %s", exc)
        return {
            "title": "Extracted Recipe",
            "raw_ingredient_lines": [],
            "instructions": raw_content[:2000] if raw_content else None,
        }

    if result.get("meal_type") not in _ALLOWED_MEAL_TYPES:
        result["meal_type"] = None
    if result.get("difficulty") not in _ALLOWED_DIFFICULTY:
        result["difficulty"] = None

    if not isinstance(result.get("raw_ingredient_lines"), list):
        result["raw_ingredient_lines"] = []

    log.info(
        "Extracted recipe: '%s' with %d ingredients",
        result.get("title", "?"),
        len(result.get("raw_ingredient_lines", [])),
    )
    return result


def agent_extract_recipe_from_image(image_base64: str, mime_type: str = "image/jpeg") -> dict:
    """
    Extracts a recipe from a base64-encoded image using Gemini vision.
    Returns the same structure as agent_extract_recipe().

    Args:
        image_base64: Base64-encoded image data (not the data URI prefix).
        mime_type: The image MIME type, e.g. "image/jpeg", "image/png".

    WO#16 update: this function now routes through
    services.ai.call_gemini_vision_json() instead of building its own
    inlineData payload and calling post_with_retry directly. The vision
    payload shape (systemInstruction + inlineData + text part +
    responseSchema) and the 120s timeout are both preserved exactly —
    see call_gemini_vision_json()'s own docstring in
    services/ai/providers/gemini.py for why 120s specifically, and the
    WO#16 postmortem for the mocked-request-shape verification. This
    closes the vision-support gap flagged in the WO#12 postmortem
    (Part 4, item 4b) and WO#15 postmortem (§6.6) — this function no
    longer imports post_with_retry or get_provider_key directly; both of
    those direct-import exceptions (introduced in WO#12's Amendment 6)
    are now retired in favor of the real service-layer function they
    were always meant to be a stopgap for.
    """
    try:
        raw = call_gemini_vision_json(
            system=_EXTRACTION_SYSTEM,
            image_base64=image_base64,
            mime_type=mime_type,
            prompt="Extract the recipe from this image.",
            schema=_EXTRACTION_SCHEMA,
            model=MODEL_FLASH,
        )
        result = _safe_json(raw)
    except Exception as exc:
        log.error("Image recipe extraction failed: %s", exc)
        return {"title": "Recipe from Image", "raw_ingredient_lines": []}

    if result.get("meal_type") not in _ALLOWED_MEAL_TYPES:
        result["meal_type"] = None
    if result.get("difficulty") not in _ALLOWED_DIFFICULTY:
        result["difficulty"] = None
    if not isinstance(result.get("raw_ingredient_lines"), list):
        result["raw_ingredient_lines"] = []

    return result


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — RECIPE DISCOVERER
# Provider: Gemini 2.5 Flash
# Frequency: On-demand from /recipes/discover
#
# Two discovery modes:
#   Pantry mode: given a list of available ingredients, suggest recipes.
#   Open mode: given mood/preferences, suggest recipes from scratch.
# ─────────────────────────────────────────────────────────────────────────────

_DISCOVERY_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "title":               {"type": "STRING"},
            "cuisine":             {"type": "STRING"},
            "meal_type":           {"type": "STRING"},
            "prep_time_minutes":   {"type": "INTEGER"},
            "cook_time_minutes":   {"type": "INTEGER"},
            "servings":            {"type": "INTEGER"},
            "difficulty":          {"type": "STRING"},
            "description":         {"type": "STRING"},
            "why_it_fits":         {"type": "STRING"},
            "key_ingredients":     {"type": "ARRAY", "items": {"type": "STRING"}},
            "raw_ingredient_lines":{"type": "ARRAY", "items": {"type": "STRING"}},
            "instructions":        {"type": "STRING"},
        },
        "required": ["title", "description", "raw_ingredient_lines", "instructions"],
    },
}

_DISCOVERY_SYSTEM = """You are a creative recipe suggester helping someone decide what to cook.

For each recipe you suggest:
- title: Clear recipe name.
- cuisine: Country/style of cuisine.
- meal_type: breakfast, lunch, dinner, snack, dessert, side, drink, or other.
- prep_time_minutes: Realistic prep time in minutes.
- cook_time_minutes: Realistic cook time in minutes.
- servings: Number of servings.
- difficulty: easy, medium, or hard.
- description: 2-3 sentence description that makes this recipe sound appealing.
- why_it_fits: 1 sentence explaining why this recipe fits the user's request.
- key_ingredients: A short list (3-5) of the most important ingredients.
- raw_ingredient_lines: COMPLETE ingredient list as strings, e.g. ["2 cups flour", "1 tsp salt"].
- instructions: Full numbered Markdown steps. Don't be lazy — include all steps.

Be specific and practical. Suggest real recipes with real techniques.
Vary the suggestions — don't suggest 5 pasta dishes when asked for dinner ideas.
"""


def agent_discover_recipes_pantry(
    pantry_ingredients: list[str],
    mood: str = "",
    meal_type: str = "",
    servings: int = 0,
    occasion: str = "",
) -> list[dict]:
    """
    Suggests recipes based on available pantry ingredients.
    """
    pantry_block = "\n".join(f"- {ing}" for ing in pantry_ingredients)
    context_parts = []
    if mood:         context_parts.append(f"Mood/craving: {mood}")
    if meal_type:    context_parts.append(f"Meal type: {meal_type}")
    if servings > 0: context_parts.append(f"Servings needed: {servings}")
    if occasion:     context_parts.append(f"Occasion: {occasion}")
    context_block = "\n".join(context_parts) if context_parts else "No specific preference."

    prompt = (
        f"Suggest 5 recipes the user can make with these pantry ingredients.\n\n"
        f"Available ingredients:\n{pantry_block}\n\n"
        f"Preferences:\n{context_block}\n\n"
        f"The user may not have every single ingredient — suggest recipes where "
        f"the pantry covers most of the key ingredients. Flag any important "
        f"missing ingredient in the why_it_fits field."
    )

    try:
        raw = call_gemini_json(prompt, schema=_DISCOVERY_SCHEMA, system=_DISCOVERY_SYSTEM)
        results = _safe_json(raw)
    except Exception as exc:
        log.error("Pantry discovery failed: %s", exc)
        return []

    return _validate_discovery_results(results)


def agent_discover_recipes_open(
    mood: str = "",
    meal_type: str = "",
    servings: int = 0,
    occasion: str = "",
    dietary_restrictions: str = "",
    cuisine_preference: str = "",
) -> list[dict]:
    """
    Suggests recipes based on preferences alone (no pantry required).
    """
    context_parts = []
    if mood:                  context_parts.append(f"Mood/craving: {mood}")
    if meal_type:             context_parts.append(f"Meal type: {meal_type}")
    if servings > 0:          context_parts.append(f"Servings needed: {servings}")
    if occasion:              context_parts.append(f"Occasion: {occasion}")
    if dietary_restrictions:  context_parts.append(f"Dietary restrictions: {dietary_restrictions}")
    if cuisine_preference:    context_parts.append(f"Cuisine preference: {cuisine_preference}")

    if not context_parts:
        context_parts.append("No specific preference — surprise me with something interesting.")

    prompt = (
        f"Suggest 5 diverse, interesting recipes based on these preferences:\n\n"
        + "\n".join(context_parts)
        + "\n\nMake each suggestion genuinely different from the others. "
          "Include at least one simple recipe and at least one that's more impressive."
    )

    try:
        raw = call_gemini_json(prompt, schema=_DISCOVERY_SCHEMA, system=_DISCOVERY_SYSTEM)
        results = _safe_json(raw)
    except Exception as exc:
        log.error("Open discovery failed: %s", exc)
        return []

    return _validate_discovery_results(results)


def _validate_discovery_results(results: list) -> list[dict]:
    """Validates and sanitizes discovery results against allowed enum values."""
    validated = []
    for item in results:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        if item.get("meal_type") not in _ALLOWED_MEAL_TYPES:
            item["meal_type"] = None
        if item.get("difficulty") not in _ALLOWED_DIFFICULTY:
            item["difficulty"] = None
        if not isinstance(item.get("raw_ingredient_lines"), list):
            item["raw_ingredient_lines"] = []
        if not isinstance(item.get("key_ingredients"), list):
            item["key_ingredients"] = []
        validated.append(item)
    return validated
