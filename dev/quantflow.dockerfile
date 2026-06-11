# Multi-stage build for quantflow app
# Stage 1: Build stage
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /build

# Install Node.js for Observable Framework frontend build
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock readme.md ./

# Install dependencies (no root package, with needed extras)
RUN uv sync --frozen --no-install-project --group docs --extra data

# Copy source and build docs
# Example outputs and images must be prebuilt in the build context
# (run `uv run ./dev/build-examples` locally, or the build-examples CI job)
COPY mkdocs.yml ./
COPY dev/ ./dev/
COPY docs/ ./docs/
COPY quantflow/ ./quantflow/
COPY frontend/ ./frontend/
COPY app/ ./app/
RUN npm --prefix frontend install
RUN npm --prefix frontend run build
RUN uv run mkdocs build

# Stage 2: Runtime stage
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code (app/ from builder includes built docs)
COPY quantflow/ ./quantflow/
COPY --from=builder /build/app ./app
COPY pyproject.toml ./

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

CMD ["python", "-m", "app"]
