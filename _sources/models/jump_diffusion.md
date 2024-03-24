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

The library allows to create a vast array of jump-diffusion models. The most famous one is the Merton jump-diffusion model.

## Merton Model

```{code-cell} ipython3
from quantflow.sp.jump_diffusion import Merton

pr = Merton.create(diffusion_percentage=0.2, jump_intensity=50, jump_skew=-0.5)
pr
```

### Marginal Distribution

```{code-cell} ipython3
m = pr.marginal(0.02)
m.std(), m.std_from_characteristic()
```

```{code-cell} ipython3
from quantflow.utils import plot

plot.plot_marginal_pdf(m, 128, normal=True, analytical=False, log_y=True)
```

### Characteristic Function

```{code-cell} ipython3
plot.plot_characteristic(m)
```

### Option Pricing

We can price options using the `OptionPricer` tooling.

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

This term structure of volatility demostrates one of the principal weakness of the Merton's model, and indeed of all jump diffusion models based on Lévy processes, namely the rapid flattening of the volatility surface as time-to-maturity increases.
For very short time-to-maturities, however, the model has no problem in producing steep volatility smile and skew.

+++

### MC paths

```{code-cell} ipython3
pr.sample(20, time_horizon=1, time_steps=1000).plot().update_traces(line_width=0.5)
```

## Exponential Jump Diffusion

This is a variation of the Mertoin model, where the jump distribution is a double exponential, one for the negative jumps and one for the positive jumps.

```{code-cell} ipython3
from 
```
