---
name: quantflow-instructions
description: 'Instructions for quantflow'
applyTo: '/**'
---


# Quantflow Instructions


## Development

* Always run `make lint` after code changes — runs taplo, isort, black, ruff, and mypy
* Never edit `readme.md` directly — it is generated from `docs/index.md` via `make docs`
* To install all dependencies (including all optional extras) run `make install-dev` — runs `uv sync --all-extras`
* To run all tests use `make test` — runs all tests in the `tests/` directory using pytest
* To run a specific test file, use `uv run pytest tests/path/to/test_file.py`

## Docker

* The Dockerfile is at `dev/quantflow.dockerfile`
* Uses `ghcr.io/astral-sh/uv:python3.14-bookworm-slim` as the base image (uv + Python bundled, no separate install needed)
* Multi-stage build: builder installs deps and builds docs, runtime copies the `.venv` and app code
* Package manager is `uv` — do not use Poetry or pip directly

## Documentation

* The documentation for quantflow is available at `https://quantflow.quantmid.com`
* Documentation is built using [mkdocs](https://www.mkdocs.org/) and stored in the `docs/` directory. The documentation source files are written in markdown format.
* Split prose into short paragraphs (one idea per paragraph) separated by blank lines. Never write a wall-of-text paragraph that strings together mechanism, rationale, caveats and usage advice. This applies to mkdocs tutorials, theory pages and long docstrings.
* Do not use dashes (em dashes, en dashes, or hyphens used as dashes) in documentation files or docstrings. Use colons, parentheses, or restructure the sentence instead.
* Math in documentation and docstrings: always use `\begin{equation}...\end{equation}` for any formula or equation. Use `$...$` only for brief inline references to variables (e.g. $F$, $K$). Do not use `$$...$$`, `` `...` ``, or RST syntax (`.. math::`, `:math:`).
* Math notation convention: use $\Phi$ for the characteristic function and $\phi$ for the characteristic exponent, where $\Phi = e^{-\phi}$.
* Glossary entries in `docs/glossary.md` must be kept in alphabetical order.
* Do not repeat concept definitions inline in tutorials or docstrings — link to the glossary instead using a relative markdown link (e.g. `[moneyness](../glossary.md#moneyness)`).
* To rebuild doc examples run `uv run ./dev/build-examples` — runs all scripts in `docs/examples/` and writes their output to `docs/examples_output/`

## Pydantic models

* Always document Pydantic fields with `Field(description=...)` — never use a docstring below a field assignment
* Split long description strings across lines using implicit string concatenation rather than shortening the text
* When a docstring line exceeds the line length limit, split it across multiple lines rather than shortening the text

## Package structure

* Strategy runtime markdown descriptions (read by `load_description()` at runtime) live inside the package at `quantflow/options/strategies/docs/` — they must be inside the package to be accessible when the library is installed
* mkdocs documentation pages live in `docs/api/options/` — do not mix these two locations
