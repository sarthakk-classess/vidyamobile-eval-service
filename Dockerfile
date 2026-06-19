FROM python:3.11-slim AS base

WORKDIR /app

# System deps for scikit-learn (C extensions) and supabase (httpx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── deps layer (cached unless requirements change) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── app code ───────────────────────────────────────────────────────────────
COPY eval_service/ eval_service/
COPY routers/      routers/

# Pre-trained GBR model artifact (built by CI before image bake)
COPY eval_service/difficulty/artifacts/ eval_service/difficulty/artifacts/

EXPOSE 8002

ENV EVAL_SERVICE_HOST=0.0.0.0 \
    EVAL_SERVICE_PORT=8002 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "eval_service.main:app", "--host", "0.0.0.0", "--port", "8002"]
