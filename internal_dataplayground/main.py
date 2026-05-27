from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import init_db

# ── Routers ────────────────────────────────────────────────────────────────────
from routers import jobs, ats, staging, dashboard, blog, explorer
from routers import finance_summary, finance_ledger, finance_upload, finance_settings
from routers import ci_projects, ci_files, ci_readme
from routers import habits, journal
from routers import recipe_extract, recipe_discovery, pantry, recipes   # ← NEW
from routers import workout, workout_log, workout_plans, workout_settings
from routers import media, media_search, media_recommend, media_settings
from routers import intent, weekly_plan


templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Finance (specific prefixes before catch-all /finance summary) ──────────────
app.include_router(finance_ledger.router)
app.include_router(finance_upload.router)
app.include_router(finance_settings.router)
app.include_router(finance_summary.router)

# ── Other modules ──────────────────────────────────────────────────────────────
app.include_router(jobs.router)
app.include_router(ats.router)
app.include_router(staging.router)
app.include_router(blog.router)
app.include_router(explorer.router)
app.include_router(habits.router)
app.include_router(journal.router)

# ── Code Intelligence (files + readme before projects for path specificity) ────
app.include_router(ci_files.router)
app.include_router(ci_readme.router)
app.include_router(ci_projects.router)

app.include_router(dashboard.router)

app.include_router(recipe_extract.router)     # ← NEW — before recipes
app.include_router(recipe_discovery.router)   # ← NEW — before recipes
app.include_router(pantry.router)             # ← NEW
app.include_router(recipes.router)            # ← NEW — last, has /{id} catch-all

app.include_router(workout.router)
app.include_router(workout_log.router)
app.include_router(workout_log.body_metrics_router)   # separate sub-router!
app.include_router(workout_plans.router)
app.include_router(workout_settings.router)

app.include_router(media_search.router)      # /media/search/*
app.include_router(media_recommend.router)   # /media/recommend/*
app.include_router(media_settings.router)    # /media/settings/*
app.include_router(media.router)             # /media (catch-all last)

# In the app.include_router section:
app.include_router(intent.router)
app.include_router(weekly_plan.router)

# ── Global 500 handler ─────────────────────────────────────────────────────────

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """
    Returns a styled HTML error page for unhandled 500s instead of
    FastAPI's raw JSON response. Logs the exception for debugging.
    """
    import logging
    logging.getLogger(__name__).exception("Unhandled 500 error: %s", exc)
    return templates.TemplateResponse(
        "500.html",
        {"request": request, "detail": str(exc)},
        status_code=500,
    )

# ── Global 404 handler ─────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    """
    Returns a styled HTML error page for unhandled 404 instead of
    simple {"detail":"Not Found"} message. Logs the exception for debugging.
    """
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)


# ── Root redirect ──────────────────────────────────────────────────────────────
from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")
