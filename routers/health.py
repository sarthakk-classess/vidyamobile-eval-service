"""routers/health.py — GET /v1/health"""

from fastapi import APIRouter
from eval_service.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok", "service": "vidya-eval", "env": settings.environment}
