---
name: quantflow-instructions
description: 'Instructions for quantflow'
applyTo: '/**'
---


# Quantflow Instructions


## Development

* Always run `make lint` after code changes — runs taplo, isort, black, ruff, and mypy
* Never edit `readme.md` directly — it is generated from `docs/index.md` via `make docs`

## Docker

* The Dockerfile is at `dev/quantflow.dockerfile`
* Uses `ghcr.io/astral-sh/uv:python3.14-bookworm-slim` as the base image (uv + Python bundled, no separate install needed)
* Multi-stage build: builder installs deps and builds docs, runtime copies the `.venv` and app code
* Package manager is `uv` — do not use Poetry or pip directly

## Documentation

* The documentation for quantflow is available at `https://quantflow.quantmid.com`
* Documentation is built using [mkdocs](https://www.mkdocs.org/) and stored in the `docs/` directory. The documentation source files are written in markdown format.

