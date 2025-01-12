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

# Jump Diffusion Models

The library allows to create a vast array of jump-diffusion models. The most famous one is the Merton jump-diffusion model.

## Merton Model

```{code-cell} ipython3
from quantflow.sp.jump_diffusion import JumpDiffusion
from quantflow.utils.distributions import Normal

merton = JumpDiffusion.create(Normal, jump_fraction=0.8, jump_intensity=50)
```

### Marginal Distribution

```{code-cell} ipython3
m = merton.marginal(0.02)
m.std(), m.std_from_characteristic()
```

```{code-cell} ipython3
m2 = jd.marginal(0.02)
m2.std(), m2.std_from_characteristic()
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
pricer = OptionPricer(merton)
pricer
```

```{code-cell} ipython3
fig = None
for ttm in (0.05, 0.1, 0.2, 0.4, 0.6, 1):
    fig = pricer.maturity(ttm).plot(fig=fig, name=f"t={ttm}")
fig.update_layout(title="Implied black vols - Merton", height=500)
```

This term structure of volatility demostrates one of the principal weakness of the Merton's model, and indeed of all jump diffusion models based on Lévy processes, namely the rapid flattening of the volatility surface as time-to-maturity increases.
For very short time-to-maturities, however, the model has no problem in producing steep volatility smile and skew.

+++

### MC paths

```{code-cell} ipython3
merton.sample(20, time_horizon=1, time_steps=1000).plot().update_traces(line_width=0.5)
```

## Exponential Jump Diffusion

This is a variation of the Mertoin model, where the jump distribution is a double exponential.
The advantage of this model is that it allows for an asymmetric jump distribution, which can be useful in some cases, for example options prices with a skew.

```{code-cell} ipython3
from quantflow.utils.distributions import DoubleExponential

jd = JumpDiffusion.create(DoubleExponential, jump_fraction=0.8, jump_intensity=50, jump_asymmetry=0.2)
pricer = OptionPricer(jd)
pricer
```

```{code-cell} ipython3
fig = None
for ttm in (0.05, 0.1, 0.2, 0.4, 0.6, 1):
    fig = pricer.maturity(ttm).plot(fig=fig, name=f"t={ttm}")
fig.update_layout(title="Implied black vols - Double-exponential Jump Diffusion ", height=500)
```

```{code-cell} ipython3

```
