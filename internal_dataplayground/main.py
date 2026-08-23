from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import init_db

# ── Routers ────────────────────────────────────────────────────────────────────
from domains.habits.routers import habits # WO1
from domains.blog.routers import blog # WO2
from domains.code_intel.routers import ci_projects, ci_files, ci_readme # WO2
from domains.jobs.routers import jobs, ats, staging, job_config # WO3
from domains.explorer.routers import explorer # WO4
from domains.finance.routers import finance_summary, finance_ledger, finance_upload, finance_settings # WO5
from domains.journal.routers import journal # WO6

from domains.workout.routers import workout, workout_log, workout_plans_crud, workout_plan_ai_generator, workout_settings # WO8
from domains.media.routers import media, media_search, media_recommend, media_settings # WO9

from routers import dashboard
from routers import recipe_extract, recipe_discovery, pantry, recipes
from routers import intent, weekly_plan


templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)
# NOTE: the more specific mount must be registered before the general
# "/static" mount below — Starlette matches Mount routes in registration
# order, so registering "/static" first would greedily intercept every
# "/static/habits/*" request and 404 it against the wrong directory.
app.mount("/static/habits", StaticFiles(directory="domains/habits/static"), name="habits_static")
app.mount("/static/blog", StaticFiles(directory="domains/blog/static"), name="blog_static")
app.mount("/static/jobs", StaticFiles(directory="domains/jobs/static"), name="jobs_static")
app.mount("/static/explorer", StaticFiles(directory="domains/explorer/static"), name="explorer_static")
app.mount("/static/finance", StaticFiles(directory="domains/finance/static"), name="finance_static")
app.mount("/static/journal", StaticFiles(directory="domains/journal/static"), name="journal_static")
app.mount("/static/workout", StaticFiles(directory="domains/workout/static"), name="workout_static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Finance (specific prefixes before catch-all /finance summary) ──────────────
app.include_router(finance_ledger.router)
app.include_router(finance_upload.router)
app.include_router(finance_settings.router)
app.include_router(finance_summary.router)

# ── Other modules ──────────────────────────────────────────────────────────────
app.include_router(jobs.router)
app.include_router(job_config.router)
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
