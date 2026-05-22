---
name: quantflow-instructions
description: 'Instructions for quantflow'
applyTo: '/**'
---


# Quantflow Instructions


## Development

* Always run `make lint` after code changes — runs taplo, isort, black, ruff, and mypy
* Never edit `readme.md` directly — it is generated from `docs/index.md` via `make docs`
* To install all dependencies (including all optional extras) run `make install-dev`
* Do not modify code unless the developer explicitly asks for a code change.
* Never change code that works unless you have been asked by the developer to do so,
  or you have a good reason to believe the code is wrong.
* Concentrate on fixing the problem, not on making the code look nice unless you are
  extremely confident that your code style is better than the original and that the original code style
  is not serving a purpose (e.g. readability, consistency with other code, etc.).
  If you are unsure, ask the developer or leave the code style as it is.

## Run Tests

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
* Always use `Annotated(..., Doc("..."))` for docstrings in code, never use triple-quoted strings below the definition of a function or class. For example:
  ```python
  from typing_extensions import Annotated, Doc

  def foo(x: Annotated[int, Doc("This is the docstring for x")]) -> float:
      """This is the docstring for foo"""
      return float(x)
  ```
* Do not use Docstrings with markdown text that may genereate headings (e.g. `# Heading`, `## Heading`, etc.)
* Math in documentation and docstrings: always use `\begin{equation}...\end{equation}` for any formula or equation. Use `$...$` only for brief inline references to variables (e.g. $F$, $K$). Do not use `$$...$$`, `` `...` ``, or RST syntax (`.. math::`, `:math:`).
* Math notation convention: use $\Phi$ for the characteristic function and $\phi$ for the characteristic exponent, where $\Phi = e^{-\phi}$.
* Glossary entries in `docs/glossary.md` must be kept in alphabetical order.
* Do not repeat concept definitions inline in tutorials or docstrings, link to the glossary instead using a relative markdown link (e.g. `[moneyness](../glossary.md#moneyness)`).
* Use relative links for all mkdocs page links (e.g. `[Option Pricing](../theory/option_pricing.md)`) — prefer relative over absolute URLs to keep links shorter and portable.
* Prefer mkdocstrings relative cross-references whenever the target is visible from the current scope: write `[label][.member]` (same class) or `[label][..Sibling]` (same module) instead of repeating the fully-qualified path. Use the full path only when the target lives in a different module than the current docstring.
* To rebuild doc examples run `uv run ./dev/build-examples` — runs all scripts in `docs/examples/` and writes their output to `docs/examples_output/`

## Pydantic models

* Always document Pydantic fields with `Field(description=...)`, never use a docstring below a field assignment
* Split long description strings across lines using implicit string concatenation rather than shortening the text
* When a docstring line exceeds the line length limit, split it across multiple lines rather than shortening the text

## Package structure

* Strategy runtime markdown descriptions (read by `load_description()` at runtime) live inside the package at `quantflow/options/strategies/docs/` — they must be inside the package to be accessible when the library is installed
* mkdocs documentation pages live in `docs/api/options/` — do not mix these two locations

## Code Conventions

* Always use `utcnow` from `quantflow.utils.dates` for getting the current UTC datetime, never use `datetime.utcnow()` or `datetime.now()`
* Never import inside functions unless explicitly discussed; all imports should be at the top of the module
