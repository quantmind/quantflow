# <a href="https://quantmind.github.io/quantflow"><img src="https://raw.githubusercontent.com/quantmind/quantflow/main/docs/assets/logos/quantflow-lockup.svg" width=300 /></a>

[![PyPI version](https://badge.fury.io/py/quantflow.svg)](https://badge.fury.io/py/quantflow)
[![Python versions](https://img.shields.io/pypi/pyversions/quantflow.svg)](https://pypi.org/project/quantflow)
[![Python downloads](https://static.pepy.tech/badge/quantflow/month)](https://pepy.tech/project/quantflow)
[![build](https://github.com/quantmind/quantflow/actions/workflows/build.yml/badge.svg)](https://github.com/quantmind/quantflow/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/quantmind/quantflow/branch/main/graph/badge.svg?token=wkH9lYKOWP)](https://codecov.io/gh/quantmind/quantflow)

Quantitative analysis and pricing tools.

![btcvol](https://github.com/quantmind/quantflow/assets/144320/88ed85d1-c3c5-489c-ac07-21b036593214)

## Installation

```bash
pip install quantflow
```
### Optional dependencies

* `data` — data retrieval: `pip install quantflow[data]`
* `ai` — MCP server for AI clients: `pip install quantflow[ai,data]`
* `ml` — training the Deep Implied Volatility model: `pip install quantflow[ml]`

## Features

* **Stochastic Processes**: a library of continuous-time models including Wiener processes, Poisson jumps, CIR mean-reverting dynamics, Heston stochastic volatility, jump-diffusion models, and the Barndorff-Nielsen & Shephard (BNS) model. Each process exposes its [characteristic function](https://quantflow.quantmind.com/theory/characteristic/) for analytical pricing.

* **Option Pricing and Calibration**: Black-Scholes pricing, implied volatility surfaces, SVI parameterisation, put/call parity, and model calibration (Heston, Double Heston). Includes support for both inverse (crypto) and standard (equity) quoting conventions.

* **Interest Rates**: yield curve construction via Nelson-Siegel and Vasicek models, discount factor calculation, and rate interpolation.

* **Market Data**: connectors for [Deribit](https://www.deribit.com), [Yahoo Finance](https://finance.yahoo.com), [Financial Modeling Prep](https://financialmodelingprep.com) (FMP), [FRED](https://fred.stlouisfed.org), the [Federal Reserve](https://www.federalreserve.gov), and [US Fiscal Data](https://fiscaldata.treasury.gov) APIs.

* **Time Series Analysis**: exponentially weighted moving averages (EWMA), Kalman filtering, super-smoothers, and OHLC bar utilities.

* **AI Integration**: an [MCP server](https://quantflow.quantmind.com/mcp/) that exposes quantflow's data tools to AI assistants.

* **JSON Serializable**: all models and pricers are built on [Pydantic](https://docs.pydantic.dev), making them fully serializable to and from JSON.

## Citation

If you use Quantflow in your research, please cite it using the metadata in [CITATION.cff](https://github.com/quantmind/quantflow/blob/main/CITATION.cff).

## License

Released under the [BSD 3-Clause License](https://github.com/quantmind/quantflow/blob/main/LICENSE).
