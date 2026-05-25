Fit a yield curve model to a set of (time-to-maturity, rate) pairs and return
the calibrated curve plus interpolated rates.

Four curve types are supported:

- **nelson_siegel**: the Nelson-Siegel parametric model, well suited for smooth
  yield curves with a single hump. Good default choice.
- **cir_curve**: Cox-Ingersoll-Ross model, a mean-reverting short-rate model.
- **vasicek_curve**: Vasicek model, a Gaussian mean-reverting short-rate model.
- **no_discount**: flat discount curve (no discounting), useful as a baseline.

Rates should be continuously compounded and expressed as decimals (e.g. 0.045 for 4.5%).
Time to maturity is in years (e.g. 0.25 for 3 months, 2.0 for 2 years).

If `max_ttm` is provided, the response includes `num_points` interpolated rates
evenly spaced up to that maturity, which is useful for plotting a smooth curve.
Otherwise, rates are returned at the input maturities only.
