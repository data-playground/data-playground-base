# routers/recipe_extract.py
"""
Recipe Extraction Pipeline

Endpoints:
  GET  /recipes/extract                → Extraction landing page (3 tabs)
  POST /recipes/extract/url            → Extract from URL → preview partial
  POST /recipes/extract/file           → Extract from PDF or image → preview partial
  POST /recipes/extract/confirm        → Save previewed recipe → redirect to detail

Extraction strategy:
  URL:   1. requests.get() the page HTML
         2. Parse Schema.org/Recipe JSON-LD (covers most recipe sites)
         3. If no JSON-LD found, strip HTML tags and pass to Gemini Flash
         4. Return preview partial for user review before saving

  File:  PDF  → extract text → pass to Gemini Flash as text
         Image → base64 encode → pass to Gemini Flash vision endpoint

TODO (Playwright headless browser):
  The _fetch_url_content() function below uses requests.get() which fails
  on JS-rendered sites (NYT Cooking, Bon Appétit, Serious Eats, etc.)
  because those pages return near-empty HTML before JS executes.

  Integration point: inside _fetch_url_content(), after the requests
  attempt, add a fallback branch:

      if _looks_empty(html):
          from playwright.async_api import async_playwright
          async with async_playwright() as p:
              browser = await p.chromium.launch()
              page = await browser.new_page()
              await page.goto(url, wait_until="networkidle")
              html = await page.content()
              await browser.close()

  The rest of the pipeline (Schema.org parse → Gemini fallback) is
  unchanged — Playwright just provides better raw HTML as input.
  Install: pip install playwright && playwright install chromium
"""

import base64
import json
import logging
import re
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Recipe, RecipeDifficulty, RecipeMealType, RecipeSourceType
from services.recipe_service import run_normalization_pipeline

log = logging.getLogger(__name__)

router = APIRouter(prefix="/recipes/extract", tags=["Recipe Extraction"])
templates = Jinja2Templates(directory="templates")


# ── URL content fetcher ────────────────────────────────────────────────────────

async def _fetch_url_content(url: str) -> str:
    """
    Fetches raw HTML from a URL.
    Uses httpx with a browser-like User-Agent to avoid bot-blocking.

    TODO (Playwright): If the returned HTML appears to be JS-rendered
    (very short or missing recipe content), fall through to a Playwright
    headless browser fetch here. See module docstring for implementation.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


def _parse_schema_org(html: str) -> Optional[dict]:
    """
    Extracts Schema.org/Recipe JSON-LD from page HTML.
    This is Path A — covers AllRecipes, Food Network, BBC Good Food,
    Epicurious, Simply Recipes, and most recipe-focused sites.

    Returns a normalized dict ready for the preview template, or None
    if no Recipe schema is found.
    """
    # Find all <script type="application/ld+json"> blocks
    pattern = re.compile(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        re.DOTALL | re.IGNORECASE,
    )
    for match in pattern.finditer(html):
        try:
            data = json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            continue

        # Handle both direct @type and @graph arrays
        recipe_data = None
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "Recipe" in str(item.get("@type", "")):
                    recipe_data = item
                    break
        elif isinstance(data, dict):
            if "Recipe" in str(data.get("@type", "")):
                recipe_data = data
            elif "@graph" in data:
                for item in data["@graph"]:
                    if isinstance(item, dict) and "Recipe" in str(item.get("@type", "")):
                        recipe_data = item
                        break

        if not recipe_data:
            continue

        # Parse time strings like "PT30M", "PT1H20M"
        def parse_duration(s: any) -> Optional[int]:
            if not s:
                return None
            s = str(s)
            hours = re.search(r"(\d+)H", s)
            mins = re.search(r"(\d+)M", s)
            total = 0
            if hours:
                total += int(hours.group(1)) * 60
            if mins:
                total += int(mins.group(1))
            return total if total > 0 else None

        # Extract ingredient lines — may be strings or dicts
        raw_ingredients = []
        for ing in recipe_data.get("recipeIngredient", []):
            if isinstance(ing, str) and ing.strip():
                raw_ingredients.append(ing.strip())
            elif isinstance(ing, dict):
                name = ing.get("name") or ing.get("text") or ""
                if name.strip():
                    raw_ingredients.append(name.strip())

        # Extract instructions — may be strings, HowToStep, or HowToSection
        instructions_text = ""
        raw_instructions = recipe_data.get("recipeInstructions", [])
        if isinstance(raw_instructions, str):
            instructions_text = raw_instructions
        elif isinstance(raw_instructions, list):
            steps = []
            for i, step in enumerate(raw_instructions, 1):
                if isinstance(step, str):
                    steps.append(f"{i}. {step.strip()}")
                elif isinstance(step, dict):
                    text = step.get("text") or step.get("name") or ""
                    if text.strip():
                        steps.append(f"{i}. {text.strip()}")
            instructions_text = "\n".join(steps)

        # Parse servings — may be "4 servings", "4-6", or just "4"
        servings_raw = recipe_data.get("recipeYield", "")
        servings = None
        if servings_raw:
            nums = re.findall(r"\d+", str(servings_raw))
            if nums:
                servings = int(nums[0])

        # Find image URL — may be string, list, or ImageObject
        image_url = None
        img = recipe_data.get("image")
        if isinstance(img, str):
            image_url = img
        elif isinstance(img, list) and img:
            image_url = img[0] if isinstance(img[0], str) else img[0].get("url")
        elif isinstance(img, dict):
            image_url = img.get("url")

        log.info(
            "Schema.org extraction: '%s' with %d ingredients",
            recipe_data.get("name", "?"),
            len(raw_ingredients),
        )
        return {
            "title":                recipe_data.get("name", "").strip(),
            "cuisine":              recipe_data.get("recipeCuisine", ""),
            "meal_type":            None,  # Schema.org doesn't map cleanly to our enum
            "prep_time_minutes":    parse_duration(recipe_data.get("prepTime")),
            "cook_time_minutes":    parse_duration(recipe_data.get("cookTime")),
            "total_time_minutes":   parse_duration(recipe_data.get("totalTime")),
            "servings":             servings,
            "difficulty":           None,
            "instructions":         instructions_text,
            "notes":                recipe_data.get("description", ""),
            "image_url":            image_url,
            "raw_ingredient_lines": raw_ingredients,
            "source_type":          "url",
        }

    return None


def _strip_html(html: str) -> str:
    """
    Strips HTML tags and collapses whitespace for the Gemini fallback.
    Preserves meaningful text content.
    """
    # Remove script and style blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Extraction landing page ────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def extract_landing(request: Request):
    return templates.TemplateResponse("recipe_extract.html", {
        "request": request,
        "active_module": "recipes",
    })


# ── URL extraction ─────────────────────────────────────────────────────────────

@router.post("/url", response_class=HTMLResponse)
async def extract_from_url(
    request: Request,
    url: str = Form(...),
):
    """
    Path A: Schema.org JSON-LD (no AI, instant)
    Path B: Gemini Flash fallback on stripped page text

    Returns partials/recipe_extract_preview.html for user review.
    """
    try:
        html = await _fetch_url_content(url)
    except Exception as exc:
        log.warning("URL fetch failed for %s: %s", url, exc)
        return templates.TemplateResponse(
            "partials/recipe_extract_preview.html",
            {
                "request": request,
                "error": f"Could not fetch that URL: {exc}",
                "extracted": None,
                "source_url": url,
            },
        )

    # Path A: Schema.org
    extracted = _parse_schema_org(html)

    # Path B: Gemini fallback
    if not extracted:
        log.info("No Schema.org data found for %s — falling back to Gemini", url)
        from airflow.agents.recipe_agents import agent_extract_recipe
        page_text = _strip_html(html)
        if len(page_text) < 200:
            return templates.TemplateResponse(
                "partials/recipe_extract_preview.html",
                {
                    "request": request,
                    "error": (
                        "The page appears to be JavaScript-rendered and returned "
                        "very little content. Try copying the recipe text and using "
                        "the Manual tab instead."
                        # TODO: This message will be removed when Playwright is added.
                    ),
                    "extracted": None,
                    "source_url": url,
                },
            )
        extracted = agent_extract_recipe(page_text, source_hint="from a recipe website")
        extracted["source_type"] = "url"

    extracted["source_url"] = url
    return templates.TemplateResponse(
        "partials/recipe_extract_preview.html",
        {"request": request, "extracted": extracted, "error": None, "source_url": url},
    )


# ── File extraction (PDF or image) ────────────────────────────────────────────

@router.post("/file", response_class=HTMLResponse)
async def extract_from_file(
    request: Request,
    file: UploadFile = File(...),
):
    """
    PDF: extract text with pdfplumber → pass to Gemini Flash as text.
    Image (JPEG/PNG/WEBP): base64 encode → Gemini Flash vision endpoint.

    Returns partials/recipe_extract_preview.html for user review.
    """
    content_type = file.content_type or ""
    filename = file.filename or ""
    file_bytes = await file.read()

    extracted = None

    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        # ── PDF: extract text then use text extraction agent ──────────────────
        try:
            import pdfplumber
            import io
            text_parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            pdf_text = "\n".join(text_parts)
        except Exception as exc:
            log.warning("PDF text extraction failed: %s", exc)
            return templates.TemplateResponse(
                "partials/recipe_extract_preview.html",
                {
                    "request": request,
                    "error": f"Could not read the PDF: {exc}",
                    "extracted": None,
                    "source_url": None,
                },
            )

        from airflow.agents.recipe_agents import agent_extract_recipe
        extracted = agent_extract_recipe(pdf_text, source_hint="from a PDF cookbook")
        extracted["source_type"] = "pdf"

    elif content_type.startswith("image/") or filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        # ── Image: base64 → Gemini vision ─────────────────────────────────────
        image_b64 = base64.b64encode(file_bytes).decode("utf-8")
        mime = content_type if content_type.startswith("image/") else "image/jpeg"

        from airflow.agents.recipe_agents import agent_extract_recipe_from_image
        extracted = agent_extract_recipe_from_image(image_b64, mime_type=mime)
        extracted["source_type"] = "image"

    else:
        return templates.TemplateResponse(
            "partials/recipe_extract_preview.html",
            {
                "request": request,
                "error": "Unsupported file type. Please upload a PDF or image (JPEG, PNG, WEBP).",
                "extracted": None,
                "source_url": None,
            },
        )

    return templates.TemplateResponse(
        "partials/recipe_extract_preview.html",
        {"request": request, "extracted": extracted, "error": None, "source_url": None},
    )


# ── Confirm and save ───────────────────────────────────────────────────────────

@router.post("/confirm", response_class=HTMLResponse)
async def confirm_extraction(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Receives the reviewed/edited recipe form from the preview partial
    and saves it through the full normalization pipeline.
    Redirects to the new recipe detail page on success.
    """
    form = await request.form()

    def _get(key: str, default: str = "") -> str:
        return str(form.get(key, default)).strip()

    def _get_int(key: str) -> Optional[int]:
        v = _get(key)
        try:
            return int(v) if v else None
        except ValueError:
            return None

    meal_type_enum = None
    try:
        if _get("meal_type"):
            meal_type_enum = RecipeMealType(_get("meal_type"))
    except ValueError:
        pass

    difficulty_enum = None
    try:
        if _get("difficulty"):
            difficulty_enum = RecipeDifficulty(_get("difficulty"))
    except ValueError:
        pass

    try:
        source_type = RecipeSourceType(_get("source_type", "manual"))
    except ValueError:
        source_type = RecipeSourceType.MANUAL

    recipe = Recipe(
        title=_get("title") or "Untitled Recipe",
        source_url=_get("source_url") or None,
        source_type=source_type,
        cuisine=_get("cuisine") or None,
        meal_type=meal_type_enum,
        prep_time_minutes=_get_int("prep_time_minutes"),
        cook_time_minutes=_get_int("cook_time_minutes"),
        total_time_minutes=_get_int("total_time_minutes"),
        servings=_get_int("servings"),
        difficulty=difficulty_enum,
        instructions=_get("instructions") or None,
        notes=_get("notes") or None,
        image_url=_get("image_url") or None,
    )
    db.add(recipe)
    await db.flush()

    # Ingredient lines come from the preview form as newline-separated text
    raw_ingredients_text = _get("raw_ingredients")
    ingredient_lines = [
        line.strip()
        for line in raw_ingredients_text.split("\n")
        if line.strip()
    ]
    tag_list = [t.strip() for t in _get("tags").split(",") if t.strip()]

    await run_normalization_pipeline(db, recipe, ingredient_lines, tag_list)
    await db.commit()

    return RedirectResponse(url=f"/recipes/{recipe.id}", status_code=303)
