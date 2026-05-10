from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from database import init_db

# ── Routers ────────────────────────────────────────────────────────────────────
from routers import jobs, ats, staging, dashboard
from routers import blog
from routers import explorer

# Finance — split into four focused routers (Phase 1B)
from routers import finance_summary
from routers import finance_ledger
from routers import finance_upload
from routers import finance_settings

# Code Intelligence — split into three focused routers (Phase 1B)
from routers import ci_projects
from routers import ci_files
from routers import ci_readme


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Jobs ───────────────────────────────────────────────────────────────────────
app.include_router(jobs.router)
app.include_router(ats.router)
app.include_router(staging.router)

# ── Finance ────────────────────────────────────────────────────────────────────
# Order matters: more-specific prefixes (/finance/ledger, /finance/upload,
# /finance/settings, /finance/accounts, /finance/categories,
# /finance/transactions) must be registered before the catch-all
# /finance summary router whose GET "" matches GET /finance.
app.include_router(finance_ledger.router)
app.include_router(finance_upload.router)
app.include_router(finance_settings.router)
app.include_router(finance_summary.router)

# ── Blog ───────────────────────────────────────────────────────────────────────
app.include_router(blog.router)

# ── SQL Explorer ───────────────────────────────────────────────────────────────
app.include_router(explorer.router)

# ── Code Intelligence ──────────────────────────────────────────────────────────
# ci_files and ci_readme both use /code-intel sub-paths;
# ci_projects handles the top-level /code-intel page and project CRUD.
app.include_router(ci_files.router)
app.include_router(ci_readme.router)
app.include_router(ci_projects.router)

# ── Dashboard ──────────────────────────────────────────────────────────────────
app.include_router(dashboard.router)

# ── Root redirect ──────────────────────────────────────────────────────────────
from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")
