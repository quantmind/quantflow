---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Heston Volatility Surface

```{code-cell} ipython3
from quantflow.sp.heston import HestonJ
from quantflow.options.pricer import OptionPricer

pricer = OptionPricer(model=HestonJ.create(
    vol=0.5,
    kappa=2,
    rho=-0.3,
    sigma=0.8,
    theta=0.36,
    jump_fraction=0.3,
    jump_asymmetry=1.2
))
pricer
```

```{code-cell} ipython3
pricer.plot3d(max_moneyness_ttm=1.5, support=31).update_layout(
    height=800,
    title="Heston volatility surface",
)
```

```{code-cell} ipython3

```
