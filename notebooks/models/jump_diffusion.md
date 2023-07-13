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

# Jump Diffusion Models

The library allows to create a vast array of jump-diffusion models. The most famous one is the Merton jump diffusion model.

```{code-cell} ipython3
from quantflow.sp.jump_diffusion import Merton

pr = Merton.create(diffusion_percentage=0.2, jump_intensity=50, jump_mean=0.0)
pr
```

## Marginal Distribution

```{code-cell} ipython3
m = pr.marginal(0.02)
m.std(), m.std_from_characteristic()
```

```{code-cell} ipython3
from quantflow.utils import plot

plot.plot_marginal_pdf(m, 128, normal=True, analytical=False, log_y=True)
```

## Characteristic Function

```{code-cell} ipython3
plot.plot_characteristic(m)
```

```{code-cell} ipython3
from quantflow.options.pricer import OptionPricer
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
