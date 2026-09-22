# routers/recipe_extract.py
"""
Recipe Extraction Pipeline

Endpoints:
  GET  /recipes/extract                → Extraction landing page (3 tabs)
  POST /recipes/extract/url            → Extract from URL → preview partial
  POST /recipes/extract/file           → Extract from PDF or image → preview partial
  POST /recipes/extract/confirm        → Save previewed recipe → redirect to detail

Extraction strategy:
  URL:   1. httpx.AsyncClient fetches the page HTML
         2. Parse Schema.org/Recipe JSON-LD (covers most recipe sites)
         3. If no JSON-LD found, strip HTML tags and pass to Gemini Flash
         4. Return preview partial for user review before saving

  File:  PDF  → extract text → pass to Gemini Flash as text
         Image → base64 encode → pass to Gemini Flash vision endpoint

The pure extraction utilities (_fetch_url_content, _parse_schema_org,
_strip_html) live in _extraction_helpers.py (WO#26 Part A).

TODO (Playwright headless browser): see the module docstring of
_extraction_helpers.py — the integration point is _fetch_url_content().
"""

import base64
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.recipes.models import Recipe, RecipeDifficulty, RecipeMealType, RecipeSourceType
from domains.recipes.routers._extraction_helpers import (
    _fetch_url_content,
    _parse_schema_org,
    _strip_html,
)
from services.recipe_service import run_normalization_pipeline
from core.templating import templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/recipes/extract", tags=["Recipe Extraction"])


# ── Error response helper ───────────────────────────────────────────────────────

def _extraction_error(request: Request, message: str, source_url: Optional[str] = None):
    """
    Renders partials/recipe_extract_preview.html in its error state.
    Shared by every failure branch below (fetch failures, unreadable files,
    and AI extraction failures) so the error shape stays consistent.
    """
    return templates.TemplateResponse(
        "partials/recipe_extract_preview.html",
        {
            "request": request,
            "error": message,
            "extracted": None,
            "source_url": source_url,
        },
    )


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
        return _extraction_error(request, f"Could not fetch that URL: {exc}", source_url=url)

    # Path A: Schema.org
    extracted = _parse_schema_org(html)

    # Path B: Gemini fallback
    if not extracted:
        log.info("No Schema.org data found for %s — falling back to Gemini", url)
        from airflow.agents.recipe_agents import agent_extract_recipe
        page_text = _strip_html(html)
        if len(page_text) < 200:
            return _extraction_error(
                request,
                "The page appears to be JavaScript-rendered and returned "
                "very little content. Try copying the recipe text and using "
                "the Manual tab instead.",
                # TODO: This message will be removed when Playwright is added.
                source_url=url,
            )
        try:
            extracted = agent_extract_recipe(page_text, source_hint="from a recipe website")
        except Exception as exc:
            log.error("Gemini extraction failed for %s: %s", url, exc)
            return _extraction_error(request, f"AI extraction failed: {exc}", source_url=url)
        if not extracted:
            return _extraction_error(
                request,
                "AI extraction returned no data. Try the Manual tab instead.",
                source_url=url,
            )
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
            return _extraction_error(request, f"Could not read the PDF: {exc}")

        from airflow.agents.recipe_agents import agent_extract_recipe
        try:
            extracted = agent_extract_recipe(pdf_text, source_hint="from a PDF cookbook")
        except Exception as exc:
            log.error("Gemini extraction failed for PDF %s: %s", filename, exc)
            return _extraction_error(request, f"AI extraction failed: {exc}")
        if not extracted:
            return _extraction_error(request, "AI extraction returned no data for this PDF.")
        extracted["source_type"] = "pdf"

    elif content_type.startswith("image/") or filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        # ── Image: base64 → Gemini vision ─────────────────────────────────────
        image_b64 = base64.b64encode(file_bytes).decode("utf-8")
        mime = content_type if content_type.startswith("image/") else "image/jpeg"

        from airflow.agents.recipe_agents import agent_extract_recipe_from_image
        try:
            extracted = agent_extract_recipe_from_image(image_b64, mime_type=mime)
        except Exception as exc:
            log.error("Gemini vision extraction failed for %s: %s", filename, exc)
            return _extraction_error(request, f"AI extraction failed: {exc}")
        if not extracted:
            return _extraction_error(request, "AI extraction returned no data for this image.")
        extracted["source_type"] = "image"

    else:
        return _extraction_error(
            request,
            "Unsupported file type. Please upload a PDF or image (JPEG, PNG, WEBP).",
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
