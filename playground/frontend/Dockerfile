# ===========================================
# beanllm Playground Frontend - Multi-stage Build
# ===========================================
# Usage:
#   docker build -t beanllm-frontend .
#   docker run -p 3000:3000 beanllm-frontend
# ===========================================

# ------------------------------------------
# Stage 1: Dependencies - Install node modules
# ------------------------------------------
FROM node:20-alpine AS deps

WORKDIR /app

# Install deps only when package files change (cache layer)
COPY playground/frontend/package.json playground/frontend/pnpm-lock.yaml* ./

RUN corepack enable pnpm && \
    pnpm install --frozen-lockfile --prod=false

# ------------------------------------------
# Stage 2: Builder - Build Next.js app
# ------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY playground/frontend/ ./

# Build-time env (backend URL for API calls)
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Enable Next.js standalone output for smaller image
ENV NEXT_TELEMETRY_DISABLED=1

RUN corepack enable pnpm && \
    pnpm build

# ------------------------------------------
# Stage 3: Runtime - Minimal production image
# ------------------------------------------
FROM node:20-alpine AS runtime

LABEL maintainer="leebeanbin <wjdqlsdu388@gmail.com>"
LABEL description="beanllm Playground Frontend"
LABEL version="0.3.0"

# Security: non-root user
RUN addgroup -S beanllm && \
    adduser -S beanllm -G beanllm

WORKDIR /app

# Copy only production artifacts
COPY --from=builder --chown=beanllm:beanllm /app/.next/standalone ./
COPY --from=builder --chown=beanllm:beanllm /app/.next/static ./.next/static
COPY --from=builder --chown=beanllm:beanllm /app/public ./public

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME="0.0.0.0"

USER beanllm

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD wget -qO- http://localhost:3000/ || exit 1

EXPOSE 3000

CMD ["node", "server.js"]
