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

# ── Code Intelligence (files + readme before projects for path specificity) ────
app.include_router(ci_files.router)
app.include_router(ci_readme.router)
app.include_router(ci_projects.router)

app.include_router(dashboard.router)


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
