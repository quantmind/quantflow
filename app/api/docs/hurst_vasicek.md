Simulate a Vasicek mean-reverting process and estimate its Hurst exponent.

The Vasicek process reverts to zero at speed `kappa`. Higher values of `kappa`
produce stronger mean reversion and a lower Hurst exponent (well below 0.5).
Low values of `kappa` approach Brownian motion (Hurst near 0.5).

The Hurst exponent is estimated from both realized variance and three OHLC-based
estimators (Parkinson, Garman-Klass, Rogers-Satchell) sampled at periods of
10min, 20min, 30min, and 1h.
