---
jupytext:
  formats: ipynb,md:myst
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

# Heston Model and Option Pricing

A very important example of time-changed Lévy process useful for option pricing is the Heston model. In this model, the Lévy process is a standard Brownian motion, while the activity rate follows a [CIR process](./cir.md). The leverage effect can be accommodated by correlating the two Brownian motions as the following equations illustrate:

\begin{align}
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
pr = Heston.create(vol=0.4, kappa=4, sigma=3, rho=-0.4)
pr
```

```{code-cell} ipython3
# check that the variance CIR process is positive
pr.variance_process.is_positive
```

## Characteristic Function

```{code-cell} ipython3
from quantflow.utils import plot
m = pr.marginal(1)
plot.plot_characteristic(m)
```

The immaginary part of the characteristic function is given by the correlation coefficient.

+++

## Marginal Distribution

Here we compare the marginal distribution at a time in the future $t=1$ with a normal distribution with the same standard deviation.

```{code-cell} ipython3
# Marginal at time 1
m = pr.marginal(1)
m.std()
```

```{code-cell} ipython3
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import numpy as np

N = 128
M = 40
r = m.pdf_from_characteristic(N, M, delta_x=4/N)
n = norm.pdf(r.x, scale=m.std())
fig = px.line(x=r.x, y=r.y, markers=True)
fig.add_trace(go.Scatter(x=r.x, y=n, name="normal", line=dict()))
```

Using log scale on the y axis highlighs the probability on the tails much better

```{code-cell} ipython3
fig = px.line(x=r.x, y=r.y, markers=True, log_y=True)
fig.add_trace(go.Scatter(x=r.x, y=n, name="normal", line=dict()))
fig.show()
```

## Option pricing

```{code-cell} ipython3
import plotly.express as px
import plotly.graph_objects as go
from quantflow.options.bs import black_call
N, M = 128, 20
dx = 4/N
r = m.call_option(N, M, 0.01, alpha=0.5)
b = black_call(r.x, m.std(), 1)
fig = px.line(x=r.x, y=r.y, markers=True)
fig.add_trace(go.Scatter(x=r.x, y=b, name="normal", line=dict()))
fig.show()
```

```{code-cell} ipython3
from quantflow.options.bs import implied_black_volatility
n = len(r.x)

result = implied_black_volatility(r.x, r.y, 1, initial_sigma=m.std()*np.ones((n,)), call_put=1)
fig = px.line(x=r.x, y=result.root, markers=True, labels=dict(x="moneyness", y="implied vol"))
fig.show()
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
