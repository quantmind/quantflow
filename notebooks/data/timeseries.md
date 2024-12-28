---
jupytext:
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

## Timeseries

```{code-cell}
from quantflow.data.fmp import FMP
from quantflow.utils.plot import candlestick_plot
cli = FMP()
```

```{code-cell}
prices = await cli.prices("ethusd", frequency="")
```

```{code-cell}
candlestick_plot(prices).update_layout(height=500)
```

```{code-cell}
from quantflow.utils.df import DFutils

df = DFutils(prices).with_rogers_satchel().with_parkinson()
df
```

```{code-cell}

```
