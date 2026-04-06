# API Reference

Complete reference for all public classes, functions, and parameters in the quantflow library.

## Modules

### [Data](data/index.md)

Clients for fetching market data from external sources. Requires the optional `data` extra:

```
pip install quantflow[data]
```

| Module | Description |
|---|---|
| [Deribit](data/deribit.md) | Crypto options and futures from the Deribit exchange |
| [Financial Modeling Prep](data/fmp.md) | Equity prices, profiles, and sector data |
| [FRED](data/fred.md) | US macroeconomic time series from the St. Louis Fed |
| [Federal Reserve](data/fed.md) | Federal Reserve H.15 interest rate data |

### [Options](options/index.md)

Option pricing, volatility surface construction, and model calibration.

| Module | Description |
|---|---|
| [Black-Scholes](options/black.md) | Black-76 pricing formula and implied volatility inversion |
| [Pricer](options/pricer.md) | Model-based option pricer supporting any stochastic process |
| [Volatility Surface](options/vol_surface.md) | Build and serialise implied volatility surfaces from market data |
| [Calibration](options/calibration.md) | Calibrate Heston and Heston-jump-diffusion models to a surface |
| [Deep IV Factor Model](options/divfm.md) | Neural-network option pricing via the DIVFM architecture |

### [Stochastic Processes](sp/index.md)

Continuous-time stochastic processes used as underlying models for option pricing and simulation.

| Module | Description |
|---|---|
| [Weiner Process](sp/weiner.md) | Geometric Brownian motion (constant volatility) |
| [Heston Model](sp/heston.md) | Stochastic volatility with optional jump component (HestonJ) |
| [Jump Diffusion](sp/jump_diffusion.md) | Compound Poisson jump processes |
| [CIR Process](sp/cir.md) | Cox-Ingersoll-Ross mean-reverting process |
| [Ornstein-Uhlenbeck](sp/ou.md) | Ornstein-Uhlenbeck mean-reverting process |
| [Poisson Process](sp/poisson.md) | Homogeneous Poisson process |
| [Compound Poisson](sp/compound_poisson.md) | Poisson arrivals with a jump-size distribution |
| [Doubly Stochastic Poisson](sp/dsp.md) | Poisson process with stochastic intensity |

### [Technical Analysis](ta/index.md)

Time series filters and indicators for financial data.

| Module | Description |
|---|---|
| [EWMA](ta/ewma.md) | Exponentially weighted moving average |
| [Kalman Filter](ta/kalman.md) | Kalman filter for state estimation |
| [Supersmoother](ta/supersmoother.md) | Ehlers two-pole supersmoother filter |
| [OHLC](ta/ohlc.md) | OHLC bar utilities and resampling |
| [Paths](ta/paths.md) | Simulated path containers and statistics |

### [Utilities](utils/index.md)

Low-level building blocks used throughout the library.

| Module | Description |
|---|---|
| [Distributions](utils/distributions.md) | Jump-size distributions (Normal, DoubleExponential) |
| [Marginal 1D](utils/marginal1d.md) | Marginal distribution via characteristic function inversion |
| [Bins](utils/bins.md) | Histogram binning helpers |
| [Numbers](utils/numbers.md) | Decimal and float numeric utilities |
| [Types](utils/types.md) | Shared type aliases |
