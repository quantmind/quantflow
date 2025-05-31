---
jupytext:
  formats: ipynb,md:myst
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

# Heston Model and Option Pricing

A very important example of time-changed Lévy process useful for option pricing is the Heston model. In this model, the Lévy process is a standard Brownian motion, while the activity rate follows a [CIR process](./cir.md). The leverage effect can be accommodated by correlating the two Brownian motions as the following equations illustrate:

\begin{align}
    y_t &= x_{\tau_t} \\
    \tau_t &= \int_0^t \nu_s ds \\
    d x_t &= d w_t \\
    d \nu_t &= \kappa\left(\theta - \nu_t\right) dt + \sigma\sqrt{\nu_t} d z_t \\
    {\mathbb E}\left[d w_t d z_t\right] &= \rho dt
\end{align}

This means that the characteristic function of $y_t=x_{\tau_t}$ can be represented as

\begin{align}
    \Phi_{y_t, u} & = {\mathbb E}\left[e^{i u y_t}\right] = {\mathbb L}_{\tau_t}^u\left(\frac{u^2}{2}\right) \\
     &= e^{-a_{t,u} - b_{t,u} \nu_0}
\end{align}

```{code-cell} ipython3
from quantflow.sp.heston import Heston
pr = Heston.create(vol=0.6, kappa=2, sigma=1.5, rho=-0.1)
pr
```

```{code-cell} ipython3
# check that the variance CIR process is positive
pr.variance_process.is_positive, pr.variance_process.marginal(1).std()
```

## Characteristic Function

```{code-cell} ipython3
from quantflow.utils import plot
m = pr.marginal(0.1)
plot.plot_characteristic(m)
```

The immaginary part of the characteristic function is given by the correlation coefficient.

+++

## Marginal Distribution

Here we compare the marginal distribution at a time in the future $t=1$ with a normal distribution with the same standard deviation.

```{code-cell} ipython3
plot.plot_marginal_pdf(m, 128, normal=True, analytical=False)
```

Using log scale on the y axis highlighs the probability on the tails much better

```{code-cell} ipython3
plot.plot_marginal_pdf(m, 128, normal=True, analytical=False, log_y=True)
```

## Option pricing

```{code-cell} ipython3
from quantflow.options.pricer import OptionPricer
from quantflow.sp.heston import Heston
pricer = OptionPricer(Heston.create(vol=0.6, kappa=2, sigma=0.8, rho=-0.2))
pricer
```

```{code-cell} ipython3
import plotly.express as px
import plotly.graph_objects as go
from quantflow.options.bs import black_call

r = pricer.maturity(0.1)
b = r.black()
fig = px.line(x=r.moneyness_ttm, y=r.time_value, markers=True, title=r.name)
fig.add_trace(go.Scatter(x=r.moneyness_ttm, y=b.time_value, name=b.name, line=dict()))
fig.show()
```

```{code-cell} ipython3
fig = None
for ttm in (0.05, 0.1, 0.2, 0.4, 0.6, 1):
    fig = pricer.maturity(ttm).plot(fig=fig, name=f"t={ttm}")
fig.update_layout(title="Implied black vols", height=500)
```

## Simulation

The simulation of the Heston model is heavily dependent on the simulation of the activity rate, mainly how the behavior near zero is handled.

The code implements algorithms from {cite:p}heston-simulation

```{code-cell} ipython3
from quantflow.sp.heston import Heston
pr = Heston.create(vol=0.6, kappa=2, sigma=0.8, rho=-0.4)
pr
```

```{code-cell} ipython3
pr.sample(20, time_horizon=1, time_steps=1000).plot().update_traces(line_width=0.5)
```

```{code-cell} ipython3
import pandas as pd
from quantflow.utils import plot

paths = pr.sample(1000, time_horizon=1, time_steps=1000)
mean = dict(mean=pr.marginal(paths.time).mean(), simulated=paths.mean())
df = pd.DataFrame(mean, index=paths.time)
plot.plot_lines(df)
```

```{code-cell} ipython3
std = dict(std=pr.marginal(paths.time).std(), simulated=paths.std())
df = pd.DataFrame(std, index=paths.time)
plot.plot_lines(df)
```

```{code-cell} ipython3

```
