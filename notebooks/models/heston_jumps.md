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

# Heston Model with Jumps

The models complements the standard [Heston](./heston.md) stochastic volatility model, with the addition of a double exponential Compound Poisson process.
The Compound Poisson process adds a jump component to the Heston diffusion SDEs which control the volatility smile and skew for shorter maturities.

\begin{align}
    y_t &= x_{\tau_t} + d j_t\\
    \tau_t &= \int_0^t \nu_s ds \\
    d x_t &= d w_t \\
    d \nu_t &= \kappa\left(\theta - \nu_t\right) dt + \sigma\sqrt{\nu_t} d z_t \\
    {\mathbb E}\left[d w_t d z_t\right] &= \rho dt
\end{align}

where $j_t$ is a double exponential Compound Poisson process which adds three additional parameter to the model

* the jump intensity, which measures the expected number of jumps in a year
* the jump percentage (fraction) contribution to the total variance
* the jump asymmetry is defined as a parameter greater than 0; 1 means jump are symmetric

The jump process is independent of the other Brownian motions.

```{code-cell} ipython3
from quantflow.sp.heston import HestonJ
pr = HestonJ.exponential(
    vol=0.6,
    kappa=2,
    sigma=0.8,
    rho=-0.2,
    jump_intensity=50,
    jump_fraction=0.2,
    jump_asymmetry=1.2
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
