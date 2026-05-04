# Release Notes

This page is the source of truth for quantflow release notes. Each section
below maps to a tagged release on
[GitHub](https://github.com/quantmind/quantflow/releases). When a new tag is
pushed, the matching section is extracted by
`.github/workflows/release.yml` and published as the GitHub Release body.

## v0.8.0

Volatility-surface calibration overhaul. This release adds a two-factor BNS
model, a double-Heston model (with optional jumps), Lewis and COS pricing
methods, and reworks the calibration package layout. Several module renames
and signature changes were made along the way: see **Breaking changes** below.

### Breaking changes

**Module renames.**

- `quantflow.sp.weiner` is now `quantflow.sp.wiener` (typo fix). Update
  imports.
- `quantflow.options.calibration` is now a package, not a single module.
  Top-level imports keep working through the package `__init__.py`
  re-exports. Code reaching into the old `quantflow.options.heston_calibration`
  must switch to `quantflow.options.calibration.heston`.

**`ModelOptionPrice` field rename.** (#47)

- `ModelOptionPrice.moneyness` previously meant `log(K/F)`. It now means
  standardised moneyness `log(K/F) / sqrt(ttm)`, and the raw log-strike is
  exposed as a new field `log_strike`. Code reading `option.moneyness` and
  expecting a log-strike must switch to `option.log_strike`.
- `get_intrinsic_value(moneyness=...)` argument renamed to `log_strike=...`.

### New features

- **`BNS2`**: two-factor Barndorff-Nielsen & Shephard stochastic-volatility
  model with a single Brownian motion driving a convex combination of
  independent Gamma-OU variances and per-factor leverage. New section in the
  BNS calibration tutorial. (#54)
- **`DoubleHeston` and `DoubleHestonJ`**: two-factor Heston (with optional
  log-price jumps) and matching `DoubleHestonCalibration` /
  `DoubleHestonJCalibration`. (#46)
- **Lewis and COS option-pricing methods**: selectable via
  `OptionPricingMethod`, alongside the existing Carr-Madan / FFT path. (#47)
- **CIR tutorial** with PDF comparison example. (#49)

### Improvements and fixes

- Heston calibration convergence fixes. (#45, #49)
- BNS calibration: dedicated `BNSCalibration` class extracted, characteristic
  exponent derivation cleaned up, broader test coverage. (#50, #51)
- OU module reworked: clearer Gamma-OU API, stronger tests for moments and
  the integrated Laplace transform. (#51)
- `pricing_method_comparison` example simplified; redundant time-comparison
  code removed. (#48)

### Documentation and assets

- New logo set (favicon, lockup, marks, social banners) under
  `docs/assets/logos/`. (#53)
- New `docs/mcp.md` page covering the MCP server.
- Bibliography rebuilt from BibTeX via `docs/bib2md.py`; glossary expanded;
  mathjax tweaks for inline rendering. (#47, #49)
- Tutorial-writing instructions added at
  `.github/instructions/tutorial.instructions.md`.

[Full changelog](https://github.com/quantmind/quantflow/compare/v0.7.0...v0.8.0)
