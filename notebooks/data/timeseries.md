---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Timeseries

```{code-cell} ipython3
from quantflow.data.fmp import FMP
from quantflow.utils.plot import candlestick_plot
cli = FMP()
```

```{code-cell} ipython3
prices = await cli.prices("aapl")
```

```{code-cell} ipython3
candlestick_plot(prices).update_layout(height=500)
```

```{code-cell} ipython3

```
