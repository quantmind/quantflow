Fit a yield curve model to a set of (time-to-maturity, rate) pairs and return
the calibrated curve plus interpolated rates.

Six curve types are supported:

- **nelson_siegel_curve**: the Nelson-Siegel parametric model, well suited for
  smooth yield curves with a single hump. Good default choice.
- **cir_curve**: Cox-Ingersoll-Ross model, a mean-reverting short-rate model.
- **vasicek_curve**: Vasicek model, a Gaussian mean-reverting short-rate model.
- **interpolated_linear_curve**: piecewise linear interpolation of the zero rate
  through the input maturities.
- **interpolated_monotonic_cubic_curve**: shape-preserving monotone cubic (PCHIP)
  interpolation of the zero rate.
- **no_discount_curve**: flat discount curve (no discounting), useful as a baseline.

Rates should be continuously compounded and expressed as decimals (e.g. 0.045 for 4.5%).
Time to maturity is in years (e.g. 0.25 for 3 months, 2.0 for 2 years).

If `max_ttm` is provided, the response includes `num_points` interpolated rates
evenly spaced up to that maturity, which is useful for plotting a smooth curve.
Otherwise, rates are returned at the input maturities only.
