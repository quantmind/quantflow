---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# HestonJ Volatility Surface

Here we study the Implied volatility surface of the Heston model with jumps.
The Heston model is a stochastic volatility model that is widely used in the finance industry to price options.

```{code-cell} ipython3
from quantflow.sp.heston import HestonJ
from quantflow.utils.distributions import DoubleExponential
from quantflow.options.pricer import OptionPricer

pricer = OptionPricer(model=HestonJ.create(
    DoubleExponential,
    vol=0.5,
    kappa=2,
    rho=-0.2,
    sigma=0.8,
    jump_fraction=0.5,
    jump_asymmetry=0.2
))
pricer
```

```{code-cell} ipython3
fig = None
for ttm in (0.1, 0.5, 1):
    fig = pricer.maturity(ttm).plot(fig=fig, name=f"ttm={ttm}")
fig
```



```{code-cell} ipython3
pricer.plot3d(max_moneyness_ttm=1.5, support=31).update_layout(
    height=800,
    title="Heston volatility surface",
)
```

```{code-cell} ipython3

```
