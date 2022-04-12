---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"tags": []}

# Heston Model and Option Pricing

A very important example of time-changed Lévy process useful for option pricing is the Heston model. In this model the Lévy process is a standard Brownian motion, while the activity rate follows a CIR process. The leverage effect can be accomodated by correlating the two Brownian motions as the following equations illustrates:

\begin{aligned}
    d x_t &= d w_t \\
    d \nu_t &= \kappa\left(\theta - \nu_t\right) dt + \sigma\sqrt{\nu_t} d z_t \\
    {\mathbb E}\left[d w_t d z_t\right] &= \rho dt
\end{aligned}

This means that the characteristic function of $y_t=x_{\tau_t}$ can be represented as

\begin{aligned}
    \Phi_{y_t, u} & = {\mathbb E}\left[e^{i u y_t}\right] = {\mathbb L}_{\tau_t}^u\left(\frac{u^2}{2}\right) \\
     &= e^{-a_{t,u} - b_{t,u} \nu_0}
\end{aligned}

```{code-cell} ipython3
from quantflow.sp.heston import Heston
pr = Heston.create(vol=0.6, kappa=1.3, sigma=0.9, rho=-0.6)
# check that the variance CIR process is positive
pr.variance_process.is_positive
```

```{code-cell} ipython3
[p for p in pr.parameters]
```

## Marginal Distribution

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
M = 20
dx = 4/N
r = m.pdf_from_characteristic(N, M, dx)
n = norm.pdf(r["x"], scale=m.std())
fig = px.line(r, x="x", y="y", markers=True)
fig.add_trace(go.Scatter(x=r["x"], y=n, name="normal", line=dict()))
fig.show()
```

Using log scale on the y axis highlighs the probability on the tails much better

```{code-cell} ipython3
fig = px.line(r, x="x", y="y", markers=True, log_y=True)
fig.add_trace(go.Scatter(x=r["x"], y=n, name="normal", line=dict()))
fig.show()
```

## Option pricing

```{code-cell} ipython3
import plotly.express as px
import plotly.graph_objects as go
from quantflow.utils.bs import black_call
N, M = 128, 10
dx = 3/N
r = m.call_option(N, M, dx, alpha=0.5)
b = black_call(r["x"], m.std(), 1)
fig = px.line(r, x="x", y="y", markers=True)
fig.add_trace(go.Scatter(x=r["x"], y=b, name="normal", line=dict()))
fig.show()
```

```{code-cell} ipython3
from quantflow.utils.bs import implied_black_volatility
s = implied_black_volatility(r["x"], r["y"], 1, initial_sigma=m.std())
fig = px.line(x=r["x"], y=s, markers=True, labels=dict(x="moneyness", y="implied vol"))
fig.show()
```
