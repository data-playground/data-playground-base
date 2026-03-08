from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import init_db
from routers import jobs

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database engine
    await init_db()
    yield
    # Shutdown logic (if needed) can go here

app = FastAPI(lifespan=lifespan)

# Plug in the Jobs module
app.include_router(jobs.router)

@app.get("/")
async def root():
    return {"message": "Life OS is online. Visit /jobs to see your tracker."}
