# routers/_extraction_helpers.py
"""
Recipe Extraction — pure helper functions

Split out of recipe_extract.py (WO#26 Part A). None of these functions have
route decorators; they are extraction utilities used by the handlers in
recipe_extract.py:

  _fetch_url_content()  → raw HTML fetch (httpx)
  _parse_schema_org()   → Schema.org/Recipe JSON-LD → normalized dict (Path A)
  _strip_html()         → tag-stripped page text for the Gemini fallback (Path B)

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

import json
import logging
import re
from typing import Optional

import httpx

log = logging.getLogger(__name__)


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
