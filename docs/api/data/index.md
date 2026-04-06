# Data

The `quantflow.data` module provides async HTTP clients for fetching market data,
economic indicators, and financial reference data from external sources.

## Installation

Data fetching requires the optional `data` extra:

```
pip install quantflow[data]
```

## Sources

| Module | Description |
|---|---|
| [Deribit](deribit.md) | Crypto options, futures, and volatility surfaces from the Deribit exchange |
| [Financial Modeling Prep](fmp.md) | Equity prices, company profiles, and sector data |
| [FRED](fred.md) | US macroeconomic time series from the St. Louis Fed |
| [Federal Reserve](fed.md) | Federal Reserve H.15 selected interest rate data |

## Usage

All clients are async context managers. Use them with `async with` to ensure connections
are properly closed:

```python
from quantflow.data.deribit import Deribit

async with Deribit() as cli:
    loader = await cli.volatility_surface_loader("btc")
```
