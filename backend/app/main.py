"""
SatQuery AI — FastAPI backend entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import query, upload, geocoding
from app.services.audit_log import init_db

app = FastAPI(
    title="SatQuery AI",
    description="Agentic vision-language assistant for single, cross-modal, and bi-temporal remote-sensing imagery.",
    version="0.1.0",
)

# Dashboard runs on a different origin during dev (Vite on :5173) — wide open for prototype speed,
# tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(geocoding.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "satquery-ai-backend"}
