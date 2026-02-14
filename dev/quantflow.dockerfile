# Multi-stage build for quantflow app
# Stage 1: Build stage
FROM python:3.14-slim AS builder

# Set working directory
WORKDIR /build

# Install poetry
RUN pip install poetry

# Copy dependency files including lock file
COPY pyproject.toml poetry.lock readme.md ./

# Configure poetry to not create virtual environments and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --with book \
    --with docs --extras data --no-interaction --no-ansi

# Copy additional files needed for docs build
COPY mkdocs.yml ./
COPY docs/ ./docs/
COPY quantflow/ ./quantflow/

# Build static documentation
RUN mkdocs build

# Stage 2: Runtime stage
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY quantflow/ ./quantflow/
COPY app/ ./app/
COPY pyproject.toml ./

# Copy built documentation from builder
COPY --from=builder /build/app/docs ./app/docs

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "-m", "app"]
