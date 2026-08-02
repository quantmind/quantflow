# Tutorials

Step-by-step guides for common quantflow workflows.

| Tutorial | Description |
|---|---|
| [Option Pricing](option_pricing.md) | Price a European option with the Black-Scholes and Heston-jump-diffusion models |
| [Pricing Method Comparison](pricing_method_comparison.md) | Compare the Carr-Madan, Lewis, and COS Fourier-based methods for pricing European options from the characteristic function |
| [Volatility Surface](volatility_surface.md) | Fetch live option data, build an implied volatility surface, and extract forwards and discount factors from option prices |
| [Heston Volatility Model](heston_calibration.md) | Calibrate the Heston and Heston-jump-diffusion models to an implied volatility surface |
| [SPX Volatility Surface](spx_vol_surface.md) | Build a 3D implied volatility surface for the S&P 500 from a Yahoo Finance option chain |
| [BNS Volatility Model](bns_calibration.md) | Calibrate the Barndorff-Nielsen and Shephard stochastic-volatility model to an implied volatility surface |
| [CIR Process](cir.md) | Explore the Cox-Ingersoll-Ross process and validate its analytical PDF against the PDF recovered from the characteristic function |
| [Yield Curve Calibration from Rates](rates_kalman.md) | Fit the Vasicek (Kalman filter) and CIR (unscented Kalman filter) short-rate models to historical Treasury rates by maximum likelihood |
| [Discount Curves from Option Prices](curve_calibration.md) | Calibrate discount curves and forwards from put-call parity, and understand why a wrong forward breaks the volatility smile |
