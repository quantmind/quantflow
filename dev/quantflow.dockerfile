# Multi-stage build for quantflow app
# Stage 1: Build stage
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /build

# Install Chromium for kaleido (Plotly static image export used by docs examples)
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock readme.md ./

# Install dependencies (no root package, with needed extras)
RUN uv sync --frozen --no-install-project --extra ai --extra book --extra docs --extra data

# Copy source, generate example outputs and images, then build docs
COPY mkdocs.yml ./
COPY dev/ ./dev/
COPY docs/ ./docs/
COPY quantflow/ ./quantflow/
RUN uv run ./dev/build-examples
RUN uv run mkdocs build

# Stage 2: Runtime stage
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code
COPY quantflow/ ./quantflow/
COPY app/ ./app/
COPY pyproject.toml ./

# Copy built documentation
COPY --from=builder /build/app/docs ./app/docs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

CMD ["python", "-m", "app"]
