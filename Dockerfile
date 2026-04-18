# ============================================================
# Dockerfile — Apertus Translation API
# ============================================================
# Supports two build targets:
#   --target api   : FastAPI only (no model; expects vLLM sidecar or DEMO_MODE=true)
#   --target full  : FastAPI + model weights baked in (large image, ~20GB)
#
# For the hackathon demo, the recommended setup is docker-compose.yml
# which starts the API in DEMO_MODE and a separate vLLM container for inference.
# ============================================================

FROM python:3.10-slim AS api

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/    ./api/
COPY ui/     ./ui/
COPY data/   ./data/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

WORKDIR /app/api

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

# Environment defaults (override in docker-compose or at runtime)
ENV MODEL_PATH="swiss-ai/Apertus-8B-Instruct-2509" \
    ADAPTER_PATH="" \
    MODEL_DEVICE="auto" \
    USE_4BIT="false" \
    DEMO_MODE="false"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]