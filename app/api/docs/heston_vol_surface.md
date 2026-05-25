Generate a theoretical implied volatility surface using a stochastic volatility model.

Two models are available:

- **jd** (Jump Diffusion): a Variance-Gamma-style model with a double-exponential
  jump distribution. Controlled by `vol`, `jump_fraction`, `jump_intensity`, and
  `jump_asymmetry`. The `kappa` and `rho` parameters are ignored.

- **hj** (Heston with Jumps): the Heston stochastic volatility model augmented with
  double-exponential jumps. All parameters are used.

The surface is returned as a grid: 10 maturities evenly spaced from 0.1 to 1.0 years,
and 51 moneyness values from -0.5 to 0.5 (log-moneyness scaled by sqrt(ttm)).
Implied volatilities that cannot be computed (e.g. deep in/out of the money) are
returned as 0.
