---
name: quantflow-tutorial-instructions
description: 'Instructions for tutorial in quantflow'
applyTo: '/docs/tutorials/**,/docs/examples/**'
---



# Tutorial Instructions

## File locations

- Tutorial pages: `docs/tutorials/<name>.md`
- Example scripts: `docs/examples/<name>.py`
- Generated images: `docs/assets/examples/<name>.png`
- Script stdout captured to: `docs/examples/output/<name>.out`
- Every new tutorial must be added to the `nav` section of `mkdocs.yml` under `Tutorials`.
- Update `docs/tutorials/index.md` with a row in the summary table.

## Building

- Build a single example: `uv run python docs/examples/<name>.py`
- Build all examples and capture output: `make docs-examples`
- Preview the docs locally: `uv run mkdocs serve`

## Tutorial page structure

Each tutorial markdown file should follow this order:

1. **H1 title** — the subject, not "Tutorial on X".
2. **One-paragraph introduction** — what the tutorial demonstrates and why it is useful.
   Link to the relevant API classes using `[ClassName][fully.qualified.path]`.
3. **Sections** (H2) — cover the concept first, then show usage, then show results.
   Use H3 subsections for variants (different parameter regimes, maturities, etc.).
4. **Code section** — always the last H2. Embed the full example script with:
   ````
   ```python
   --8<-- "docs/examples/<name>.py"
   ```
   ````
   If the script prints structured output, embed it too:
   ````
   ```
   --8<-- "docs/examples/output/<name>.out"
   ```
   ````

## Example scripts

- Each script must be self-contained and runnable with `uv run python docs/examples/<name>.py`.
- Place shared helpers in `docs/examples/_utils.py` — do not duplicate utility code.
- Use `assets_path(filename)` from `_utils.py` to get the correct path when saving images.
- Use `plotly` for all charts. Save to PNG with `fig.write_image(assets_path(...), width=900, height=500)`.
  Use `width=1600, height=800` for side-by-side subplot layouts.
- When overlaying an analytical curve with a numerical result (e.g. PDF from characteristic
  function), plot the analytical result as a solid line and the numerical result as circle
  markers (`mode="markers", marker=dict(symbol="circle")`). This makes the discretization
  points visible and the two series easy to distinguish.
- Do not `print` raw numbers — emit only what belongs in the captured `.out` file.
- Scripts must produce no warnings when run cleanly (fix the root cause, e.g. avoid `x=0`
  in domains where the PDF is singular).
- The `build_examples` helper in `_utils.py` runs every non-underscore script and captures
  stdout; keep scripts idempotent and deterministic.

## Charts

- Embed images with a clickable link that opens full-size in a new tab:
  ```markdown
  [![Alt text](../assets/examples/<name>.png)](../assets/examples/<name>.png){target="_blank"}
  ```
- Write a one-sentence caption above each image explaining what the reader should observe.

## Math

Follow the math conventions in `copilot-instructions.md`:

- Use `\begin{equation}...\end{equation}` for standalone formulas.
- Use `\begin{equation}\begin{aligned}...\end{aligned}\end{equation}` for multi-line systems.
- Use `$...$` only for brief inline variable references.
- Use `\Phi` for the characteristic function and `\phi` for the characteristic exponent.

## Cross-references

- Link API symbols with `[ClassName][fully.qualified.module.ClassName]`.
- Link to theory pages with relative markdown links: `[Option Pricing](../theory/option_pricing.md)`.
- Link to the glossary for concept definitions rather than re-defining them inline.
- Link to the bibliography for external references: `[Carr-Madan](../bibliography.md#carr_madan)`.

## What not to include

- Do not explain implementation details that belong in docstrings.
- Do not reproduce equations already in the API reference — link to them instead.
- Do not add a summary or "next steps" section unless the tutorial is part of a series.
- Do not use math notation (`$...$`, `\begin{equation}`, etc.) in any heading (H1–H4).
  Math does not render in the table of contents — write headings in plain English instead
  (e.g. "Short horizon" not "Short horizon ($t = 0.5$)").
