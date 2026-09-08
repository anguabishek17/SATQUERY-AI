"""
SatQuery AI — FastAPI backend entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""
from dotenv import load_dotenv
load_dotenv()

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(geocoding.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    # Lightweight asynchronous model warm-up
    import threading
    def _warmup():
        try:
            from app.tools.object_counting import warmup_building_model
            warmup_building_model()
        except Exception:
            pass
    threading.Thread(target=_warmup, daemon=True).start()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "satquery-ai-backend"}
