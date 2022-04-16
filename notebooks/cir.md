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

# CIR process

The Cox–Ingersoll–Ross (CIR) model is a standard mean reverting square-root process used to model interest rates and stochastic variance. It takes the form

\begin{equation}
 dx_t = \kappa\left(\theta - x_t\right) dt + \sigma \sqrt{x_t} d w_t
\end{equation}

$\kappa$ is the mean reversion speed, $\theta$ the long term value of $x_t$, $\sigma$ controls the standard deviation given by $\sigma\sqrt{x_t}$ and $w_t$ is a brownian motion.

Importantly, the process remains positive if

\begin{equation}
 2 \kappa \theta > \sigma^2
\end{equation}

The model has a close-form solution for the mean and the variance

\begin{align}
{\mathbb E}[x_t] &= x_0 e^{-\kappa t} + \theta\left(1 - e^{-\kappa t}\right) \\
{\mathbb Var}[x_t] &= x_0 \frac{\sigma^2}{\kappa}\left(e^{-\kappa t} - e^{-2 \kappa t}\right) + \frac{\theta \sigma^2}{2\kappa}\left(1 - e^{-\kappa t}\right)^2 \\
\end{align}

```{code-cell} ipython3
from quantflow.sp.cir import CIR
pr = CIR(1, kappa=0.8, sigma=0.8, theta=1.2)
pr.is_positive
```

```{code-cell} ipython3
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
p = pr.sample(10, t=1, steps=100)
x = pd.DataFrame(p.data)
fig = px.line(p.data)
fig.show()
```

```{code-cell} ipython3
from notebooks.utils import chracteristic_fig
N = 128
M = 30
m = pr.marginal(0.5)
chracteristic_fig(m, N, M).show()
```

## Computed PDF and analytical

The code below show the computed PDF via FRFT and the [analytical formula](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model).

```{code-cell} ipython3
dx = 4/N
r = m.pdf_from_characteristic(N, M, dx)
fig = go.Figure()
fig.add_trace(go.Scatter(x=r["x"], y=r["y"], mode="markers", name="computed"))
fig.add_trace(go.Scatter(x=r["x"], y=m.pdf(r["x"]), name="analytical", line=dict()))
fig.show()
```

```{code-cell} ipython3
import plotly
plotly.__version__
```

```{code-cell} ipython3

```
