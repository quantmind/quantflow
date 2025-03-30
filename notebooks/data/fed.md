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

# Federal Reserve Data

```{code-cell} ipython3
from quantflow.data.fed import FederalReserve
```

```{code-cell} ipython3
async with FederalReserve() as fed:
    rates = await fed.ref_rates()
```

```{code-cell} ipython3
rates
```

```{code-cell} ipython3
async with FederalReserve() as fed:
    curves = await fed.yield_curves()
```

```{code-cell} ipython3
curves
```

```{code-cell} ipython3

```
