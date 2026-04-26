from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from database import init_db
from routers import jobs, ats, staging, finance, blog, explorer, code_intelligence, dashboard

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database engine
    await init_db()
    yield
    # Shutdown logic (if needed) can go here

app = FastAPI(lifespan=lifespan)

# Serve static files (CSS, JS) — required for Option C base.html
app.mount("/static", StaticFiles(directory="static"), name="static")

# Plug in the Jobs module
app.include_router(jobs.router)
app.include_router(ats.router)
app.include_router(staging.router)
app.include_router(finance.router)
app.include_router(blog.router)
app.include_router(explorer.router)
app.include_router(code_intelligence.router)
app.include_router(dashboard.router)

from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")
