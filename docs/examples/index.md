# Interactive Examples

Interactive notebooks for exploring quantflow tools and models.

Each example is a [marimo](https://marimo.io) notebook served as a live application.
You can run the code, adjust parameters, and see results update in real time.

!!! warning "Work in progress"
    These notebooks are not always stable and may fail to load or produce unexpected results.
    We are actively working on improving their reliability.
    If you have experience with marimo and would like to help, contributions are very welcome via [GitHub](https://github.com/quantmind/quantflow).

## Stochastic Processes

| Example | Description |
|---|---|
| [Gaussian Sampling](gaussian-sampling) | Sample the Gaussian Ornstein-Uhlenbeck (Vasicek) process for different mean-reversion speeds and path counts |
| [Poisson Sampling](poisson-sampling) | Compare Monte Carlo simulation of the Poisson process against the analytical PDF |
| [Double Exponential Sampling](double-exponential-sampling) | Explore the Asymmetric Laplace distribution with adjustable asymmetry parameter |

## Time Series Analysis

| Example | Description |
|---|---|
| [Hurst Exponent](hurst) | Estimate the Hurst exponent to classify a time series as trending, mean-reverting, or random |
| [Supersmoother](supersmoother) | Apply the Supersmoother and EWMA filters to financial time series |

## Volatility and Options

| Example | Description |
|---|---|
| [Volatility Surface](volatility-surface) | Build and visualise an implied volatility surface from live Deribit ETH/USD options data |
