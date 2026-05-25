# Release Notes

This page is the source of truth for quantflow release notes. Each section
below maps to a tagged release on
[GitHub](https://github.com/quantmind/quantflow/releases). When a new tag is
pushed, the matching section is extracted by
`.github/workflows/release.yml` and published as the GitHub Release body.

## v0.9.0

Pricing-engine and calibration overhaul. `MaturityPricer` now evaluates call
prices and Greeks lazily at arbitrary log-strikes instead of carrying a
precomputed grid, Fourier pricers take a moneyness-based truncation parameter,
and the volatility-surface calibration can fit Black implied vols directly.
This release contains several API changes: see **Breaking changes** below.

### Breaking changes

**`MaturityPricer` reworked.** ([#59](https://github.com/quantmind/quantflow/pull/59))

- The precomputed `std`, `log_strike` and `call` arrays are gone. A
  `MaturityPricer` now holds a single `pricing` field (an
  `OptionPricingResult`) that evaluates call prices and Greeks on demand at
  any log-strike.
- `moneyness` is now a method, `moneyness(log_strikes)`, not a cached array
  property. The `time_value` and `intrinsic_value` array properties and the
  `interp(...)` helper were removed; use `prices(log_strikes)` to get a
  DataFrame of prices and implied vols on a chosen log-strike grid.

**Fourier pricing truncation: `max_log_strike` → moneyness parameters.**
([#59](https://github.com/quantmind/quantflow/pull/59))

- `Marginal1D.call_option`, `call_option_carr_madan` and `call_option_lewis`
  take `max_moneyness` (a multiple of the marginal standard deviation)
  instead of `max_log_strike`. The COS path takes
  `cos_moneyness_std_precision` instead.
- `OptionPricingResult.call_at(...)` is renamed `call_price(...)`, the
  `method` field is removed, and a new abstract `call_greeks(log_strike)`
  returns a `Greeks` namedtuple `(price, delta, gamma)`.

**`OptionPricerBase.call_price` → `call_prices`.**
([#59](https://github.com/quantmind/quantflow/pull/59)) The method is now
vectorised: it takes arrays of times-to-maturity and log-strikes and prices
them in a single maturity-grouped call.

**`DIVFMPricer` no longer builds a fixed moneyness grid.**
([#59](https://github.com/quantmind/quantflow/pull/59)) The
`max_moneyness_ttm` and `n` fields are removed; the fitted IV surface is
evaluated on demand through `OptionPricingResultDIVFM`.

### New features

- **Implied-vol calibration residuals.** New `ResidualKind` enum and a
  `residual_kind` field on `VolModelCalibration`: set it to `ResidualKind.IV`
  to fit the model to Black implied vols (recovered by inverting the model
  price) rather than to forward-space prices. The IV residual is naturally
  well-scaled across moneyness, so `moneyness_weight` is not applied in that
  mode. ([#59](https://github.com/quantmind/quantflow/pull/59))
- **Greeks from the pricing result.** `OptionPricingCosResult.call_greeks`
  returns closed-form price, delta and gamma from the COS expansion; the
  transform-based result derives delta and gamma by differentiating the call
  grid; DIVFM uses finite differences on the fitted surface.
  ([#59](https://github.com/quantmind/quantflow/pull/59))
- **COS truncation control on `OptionPricer`.** New
  `cos_moneyness_std_precision` field (default 12) sets the width of the COS
  integration interval in standard deviations.
  ([#59](https://github.com/quantmind/quantflow/pull/59))

### Improvements and fixes

- Calibration residuals are now computed in a single vectorised pricing call.
  Deep-wing strikes where the model price falls outside the no-arbitrage band
  (so Newton fails to invert it) are masked out instead of poisoning the fit,
  and a parameter set that fails to invert on more than half the options is
  rejected with a large penalty.
  ([#59](https://github.com/quantmind/quantflow/pull/59))
- Calibration plots now evaluate the model on a fresh moneyness grid;
  `plot(max_moneyness=...)` no longer accepts `None`.
  ([#59](https://github.com/quantmind/quantflow/pull/59))
- `OptionEntry.mid_price()` no longer caches through a private attribute.
  ([#59](https://github.com/quantmind/quantflow/pull/59))
- Stale Jupytext notebook mirrors under `notebooks/` removed.
  ([#59](https://github.com/quantmind/quantflow/pull/59))

### Documentation and assets

- New GitHub social-preview banner under `docs/assets/logos/png/`.
  ([#59](https://github.com/quantmind/quantflow/pull/59))
- `docs/api/options/black.md` and the volatility-surface calibration examples
  updated for the new pricer API.
  ([#59](https://github.com/quantmind/quantflow/pull/59))
- The release procedure moved out of `.github/copilot-instructions.md` into
  its own `.github/instructions/release.instructions.md`.

[Full changelog](https://github.com/quantmind/quantflow/compare/v0.8.0...v0.9.0)

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

**`ModelOptionPrice` field rename.** ([#47](https://github.com/quantmind/quantflow/pull/47))

- `ModelOptionPrice.moneyness` previously meant `log(K/F)`. It now means
  standardised moneyness `log(K/F) / sqrt(ttm)`, and the raw log-strike is
  exposed as a new field `log_strike`. Code reading `option.moneyness` and
  expecting a log-strike must switch to `option.log_strike`.
- `get_intrinsic_value(moneyness=...)` argument renamed to `log_strike=...`.

### New features

- **`BNS2`**: two-factor Barndorff-Nielsen & Shephard stochastic-volatility
  model with a single Brownian motion driving a convex combination of
  independent Gamma-OU variances and per-factor leverage. New section in the
  BNS calibration tutorial. ([#54](https://github.com/quantmind/quantflow/pull/54))
- **`DoubleHeston` and `DoubleHestonJ`**: two-factor Heston (with optional
  log-price jumps) and matching `DoubleHestonCalibration` /
  `DoubleHestonJCalibration`. ([#46](https://github.com/quantmind/quantflow/pull/46))
- **Lewis and COS option-pricing methods**: selectable via
  `OptionPricingMethod`, alongside the existing Carr-Madan / FFT path.
  ([#47](https://github.com/quantmind/quantflow/pull/47))
- **CIR tutorial** with PDF comparison example.
  ([#49](https://github.com/quantmind/quantflow/pull/49))

### Improvements and fixes

- Heston calibration convergence fixes.
  ([#45](https://github.com/quantmind/quantflow/pull/45),
  [#49](https://github.com/quantmind/quantflow/pull/49))
- BNS calibration: dedicated `BNSCalibration` class extracted, characteristic
  exponent derivation cleaned up, broader test coverage.
  ([#50](https://github.com/quantmind/quantflow/pull/50),
  [#51](https://github.com/quantmind/quantflow/pull/51))
- OU module reworked: clearer Gamma-OU API, stronger tests for moments and
  the integrated Laplace transform.
  ([#51](https://github.com/quantmind/quantflow/pull/51))
- `pricing_method_comparison` example simplified; redundant time-comparison
  code removed. ([#48](https://github.com/quantmind/quantflow/pull/48))

### Documentation and assets

- New logo set (favicon, lockup, marks, social banners) under
  `docs/assets/logos/`. ([#53](https://github.com/quantmind/quantflow/pull/53))
- Bibliography rebuilt from BibTeX via `docs/bib2md.py`; glossary expanded;
  mathjax tweaks for inline rendering.
  ([#47](https://github.com/quantmind/quantflow/pull/47),
  [#49](https://github.com/quantmind/quantflow/pull/49))
- Tutorial-writing instructions added at
  `.github/instructions/tutorial.instructions.md`.

[Full changelog](https://github.com/quantmind/quantflow/compare/v0.7.0...v0.8.0)
