---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Data

The library provides a python client for the [Financial Modelling Prep API](https://site.financialmodelingprep.com/developer/docs). To use the client one needs to provide the API key aither directly to the client or via the `FMP_API_KEY` environment variable. The API offers 1 minute, 5 minutes, 15 minutes, 30 minutes, 1 hour and daily historical prices.

```{code-cell}
from quantflow.data.fmp import FMP
cli = FMP()
cli.url
```

```{code-cell}
stock = "KNOS.L"
```

## Company Profile

```{code-cell}
d = await cli.profile(stock)
d
```

```{code-cell}
c = await cli.peers(stock)
c
```

## Executive trading

```{code-cell}
stock = "KNOS.L"
```

```{code-cell}
await cli.executives(stock)
```

```{code-cell}
await cli.insider_trading(stock)
```

## News

```{code-cell}
c = await cli.news(stock)
c
```
