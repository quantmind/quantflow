Simulate a Wiener (Brownian motion) process over one day at one-second resolution
and estimate its Hurst exponent using multiple methods.

The Hurst exponent measures long-range dependence in a time series.
A value of 0.5 indicates pure Brownian motion (no memory).
Values above 0.5 indicate trending behaviour; values below 0.5 indicate mean reversion.

Three OHLC-based range estimators are computed across multiple sampling periods:

- **Parkinson**: uses high-low range.
- **Garman-Klass**: uses open, high, low, close.
- **Rogers-Satchell**: drift-adjusted version of Garman-Klass.

The Hurst exponent for each estimator is derived from the slope of log(variance)
vs log(sampling period).
