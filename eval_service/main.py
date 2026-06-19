"""
eval_service/main.py — FastAPI entry point for the Vidya eval service.

Exposes SK-01 through SK-07 capabilities as HTTP endpoints.
SK-08/10/11 eval harnesses are CLI scripts in eval/ — not HTTP endpoints,
since they run as CI jobs, not as online inference.

Start locally:
    uvicorn eval_service.main:app --reload --port 8002

Production (Docker):
    docker-compose up eval-service
"""

from __future__ import annotations
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from eval_service.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Vidya Eval Service",
    description="Chunking, embedding, mastery scheduling, and evaluation gates for Vidya AI Tutoring.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

from routers.health  import router as health_router
from routers.chunk   import router as chunk_router
from routers.embed   import router as embed_router
from routers.mastery import router as mastery_router

app.include_router(health_router,  prefix="/v1")
app.include_router(chunk_router,   prefix="/v1")
app.include_router(embed_router,   prefix="/v1")
app.include_router(mastery_router, prefix="/v1")


@app.on_event("startup")
async def _startup():
    log.info(f"Vidya eval service starting — env={settings.environment} port={settings.port}")
