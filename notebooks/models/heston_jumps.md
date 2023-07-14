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

# Heston Model with Jumps

```{code-cell} ipython3
from quantflow.sp.heston import HestonJ
pr = HestonJ.create(
    vol=0.6,
    kappa=2,
    sigma=1.5,
    rho=-0.3,
    jump_intensity=100,
    jump_fraction=0.3,
    jump_skew=1.5
)
pr
```

```{code-cell} ipython3
from quantflow.utils import plot
plot.plot_marginal_pdf(pr.marginal(0.1), 128, normal=True, analytical=False)
```

```{code-cell} ipython3
from quantflow.options.pricer import OptionPricer
from quantflow.sp.heston import Heston
pricer = OptionPricer(pr)
pricer
```

```{code-cell} ipython3
fig = None
for ttm in (0.05, 0.1, 0.2, 0.4, 0.6, 1):
    fig = pricer.maturity(ttm).plot(fig=fig, name=f"t={ttm}")
fig.update_layout(title="Implied black vols", height=500)
```

```{code-cell} ipython3

```
