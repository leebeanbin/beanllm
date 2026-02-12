# ===========================================
# beanllm Playground Backend - Multi-stage Build
# ===========================================
# Usage:
#   docker build -t beanllm-backend .
#   docker run -p 8000:8000 --env-file .env beanllm-backend
# ===========================================

# ------------------------------------------
# Stage 1: Builder - Install Python deps
# ------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install beanllm with web extras from the project root
COPY pyproject.toml README.md /build/
COPY src/ /build/src/

# Install into a virtual env for easy copying
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "/build[web,distributed]"

# Install additional backend-specific deps
COPY playground/backend/requirements-docker.txt /build/playground/backend/
RUN if [ -f /build/playground/backend/requirements-docker.txt ]; then \
        pip install --no-cache-dir -r /build/playground/backend/requirements-docker.txt; \
    fi

# ------------------------------------------
# Stage 2: Runtime - Minimal production image
# ------------------------------------------
FROM python:3.12-slim AS runtime

# Labels
LABEL maintainer="leebeanbin <wjdqlsdu388@gmail.com>"
LABEL description="beanllm Playground Backend API"
LABEL version="0.3.0"

# Security: create non-root user
RUN groupadd -r beanllm && \
    useradd -r -g beanllm -d /app -s /sbin/nologin beanllm

# Runtime-only system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        tini && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app"

WORKDIR /app

# Copy backend source code
COPY playground/backend/ /app/

# Create dirs for uploads and chroma
RUN mkdir -p /app/uploads /app/chroma_db && \
    chown -R beanllm:beanllm /app

# Switch to non-root user
USER beanllm

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

# Use tini as init process (proper PID 1 signal handling)
ENTRYPOINT ["tini", "--"]

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log"]
