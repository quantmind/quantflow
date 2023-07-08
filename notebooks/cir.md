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

# CIR process

The Cox–Ingersoll–Ross (CIR) model is a standard mean reverting square-root process used to model interest rates and stochastic variance. It takes the form

\begin{equation}
 dx_t = \kappa\left(\theta - x_t\right) dt + \sigma \sqrt{x_t} d w_t
\end{equation}

$\kappa$ is the mean reversion speed, $\theta$ the long term value of $x_t$, $\sigma$ controls the standard deviation given by $\sigma\sqrt{x_t}$ and $w_t$ is a brownian motion.

Importantly, the process remains positive if the Feller condition is satisfied

\begin{equation}
 2 \kappa \theta > \sigma^2
\end{equation}

In the code, the initial value of the process, ${\bf x}_0$, is given by the `rate` field, for example, a CIR process can be created via 

```{code-cell} ipython3
from quantflow.sp.cir import CIR
pr = CIR(rate=1.0, kappa=2.0, sigma=1.2)
pr
```

```{code-cell} ipython3
pr.is_positive
```

## Marginal and moments

The model has a closed-form solution for the mean and the variance

\begin{align}
{\mathbb E}[x_t] &= x_0 e^{-\kappa t} + \theta\left(1 - e^{-\kappa t}\right) \\
{\mathbb Var}[x_t] &= x_0 \frac{\sigma^2}{\kappa}\left(e^{-\kappa t} - e^{-2 \kappa t}\right) + \frac{\theta \sigma^2}{2\kappa}\left(1 - e^{-\kappa t}\right)^2 \\
\end{align}

```{code-cell} ipython3
m = pr.marginal(0.5)
m.mean(), m.variance()
```

```{code-cell} ipython3
m.mean_from_characteristic(), m.variance_from_characteristic()
```

The code below show the computed PDF via FRFT and the analytical formula above

```{code-cell} ipython3
import plotly.graph_objects as go
N = 128*8
M = 20
dx = 8/N
r = m.pdf_from_characteristic(N, M, dx)
fig = go.Figure()
fig.add_trace(go.Scatter(x=r["x"], y=r["y"], mode="markers", name="computed"))
fig.add_trace(go.Scatter(x=r["x"], y=m.pdf(r["x"]), name="analytical", line=dict()))
fig.show()
```

## Characteristic Function

For this process it is possible to obtain the analytical formula of $a$ and $b$:

\begin{equation}
a =-\frac{2\kappa\theta}{\sigma^2} \log{\left(\frac{c + d e^{-\gamma \tau}}{c + d}\right)} + \frac{\kappa \theta \tau}{c}\\
b = \frac{1-e^{-\gamma \tau}}{c + d e^{-\gamma_u \tau}}
\end{equation}

with
\begin{equation}
\gamma = \sqrt{\kappa^2 - 2 u \sigma^2} \\
c = \frac{\gamma + \kappa}{2 u} \\
d = \frac{\gamma - \kappa}{2 u}
\end{equation}

```{code-cell} ipython3
from notebooks.utils import chracteristic_fig
m = pr.marginal(0.5)
chracteristic_fig(m, N, M).show()
```

## Sampling

The code offers three sampling algorithms, both guarantee positiveness even if the Feller condition above is not satisfied.

The first sampling algorithm is the explicit Euler *full truncation* algorithm where the process is allowed to go below zero, at which point the process becomes deterministic with an upward drift of $\kappa \theta$, see {cite:p}`heston-calibration` and {cite:p}`heston-simulation` for a detailed discussion.

```{code-cell} ipython3
from quantflow.sp.cir import CIR
pr = CIR(rate=1.0, kappa=1.0, sigma=2.0, sample_algo="euler")
pr
```

```{code-cell} ipython3
pr.is_positive
```

```{code-cell} ipython3
pr.paths(20, t=1, steps=1000).plot().update_traces(line_width=0.5)
```

The second sampling algorithm is the implicit Milstein scheme, a refinement of the Euler scheme produced by adding an extra term using the Ito's lemma.

The third algorithm is a fully implicit one that guarantees positiveness of the process if the Feller condition is met.

```{code-cell} ipython3
pr = CIR(rate=1.0, kappa=1.0, sigma=0.8)
pr
```

```{code-cell} ipython3
pr.paths(20, t=1, steps=1000).plot().update_traces(line_width=0.5)
```


Sampling with a mean reversion speed 20 times larger

```{code-cell} ipython3
pr.kappa = 20; pr
```

```{code-cell} ipython3
pr.paths(20, t=1, steps=1000).plot().update_traces(line_width=0.5)
```

```{code-cell} ipython3

```
