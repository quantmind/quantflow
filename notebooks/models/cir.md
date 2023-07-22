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

The model has a closed-form solution for the mean, the variance, and the [marginal pdf](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model).

\begin{align}
{\mathbb E}[x_t] &= x_0 e^{-\kappa t} + \theta\left(1 - e^{-\kappa t}\right) \\
{\mathbb Var}[x_t] &= x_0 \frac{\sigma^2}{\kappa}\left(e^{-\kappa t} - e^{-2 \kappa t}\right) + \frac{\theta \sigma^2}{2\kappa}\left(1 - e^{-\kappa t}\right)^2 \\
\end{align}

```{code-cell} ipython3
m = pr.marginal(1)
m.mean(), m.variance()
```

```{code-cell} ipython3
m.mean_from_characteristic(), m.variance_from_characteristic()
```

The code below show the computed PDF via FRFT and the analytical formula above

```{code-cell} ipython3
from quantflow.utils import plot
import numpy as np
plot.plot_marginal_pdf(m, 128, max_frequency=20)
```

## Characteristic Function

For this process, it is possible to obtain the analytical formula of $a$ and $b$:

\begin{align}
a &=-\frac{2\kappa\theta}{\sigma^2} \log{\left(\frac{c + d e^{-\gamma \tau}}{c + d}\right)} + \frac{\kappa \theta \tau}{c}\\
b &= \frac{1-e^{-\gamma \tau}}{c + d e^{-\gamma_u \tau}}
\end{align}

with
\begin{align}
\gamma &= \sqrt{\kappa^2 - 2 u \sigma^2} \\
c &= \frac{\gamma + \kappa}{2 u} \\
d &= \frac{\gamma - \kappa}{2 u}
\end{align}

```{code-cell} ipython3
from quantflow.utils import plot
m = pr.marginal(0.5)
plot.plot_characteristic(m)
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
pr.sample(20, time_horizon=1, time_steps=1000).plot().update_traces(line_width=0.5)
```

The second sampling algorithm is the implicit Milstein scheme, a refinement of the Euler scheme produced by adding an extra term using the Ito's lemma.

The third algorithm is a fully implicit one that guarantees positiveness of the process if the Feller condition is met.

```{code-cell} ipython3
pr = CIR(rate=1.0, kappa=1.0, sigma=0.8)
pr
```

```{code-cell} ipython3
pr.sample(20, time_horizon=1, time_steps=1000).plot().update_traces(line_width=0.5)
```

Sampling with a mean reversion speed 20 times larger

```{code-cell} ipython3
pr.kappa = 20; pr
```

```{code-cell} ipython3
pr.sample(20, time_horizon=1, time_steps=1000).plot().update_traces(line_width=0.5)
```

## MC simulations

In this section we compare the performance of the three sampling algorithms in estimating the mean and and standard deviation.

```{code-cell} ipython3
from quantflow.sp.cir import CIR

params = dict(rate=0.8, kappa=1.5, sigma=1.2)
pr = CIR(**params)

prs = [
    CIR(sample_algo="euler", **params),
    CIR(sample_algo="milstein", **params),
    pr
]
```

```{code-cell} ipython3
import pandas as pd
from quantflow.utils import plot
from quantflow.utils.paths import Paths

samples = 1000
time_steps = 100

draws = Paths.normal_draws(samples, time_horizon=1, time_steps=time_steps)
mean = dict(mean=pr.marginal(draws.time).mean())
mean.update({pr.sample_algo.name: pr.sample_from_draws(draws).mean() for pr in prs})
df = pd.DataFrame(mean, index=draws.time)

plot.plot_lines(df)
```

```{code-cell} ipython3
std = dict(std=pr.marginal(draws.time).std())
std.update({pr.sample_algo.name: pr.sample_from_draws(draws).std() for pr in prs})
df = pd.DataFrame(std, index=draws.time)

plot.plot_lines(df)
```

## Integrated log-Laplace Transform

The log-Laplace transform of the integrated CIR process is defined as

\begin{align}
\iota_{t,u} &= \log  {\mathbb E}\left[e^{- u \int_0^t x_s ds}\right]\\
    &= a_{t,u} + x_0 b_{t,u}\\
    a_{t,u} &= \frac{2\kappa\theta}{\sigma^2} \log{\frac{2\gamma_u e^{\left(\kappa+\gamma_u\right)t/2}}{d_{t,u}}}\\
    b_{t,u} &=-\frac{2u\left(e^{\gamma_u t}-1\right)}{d_{t,u}}\\
    d_{t,u} &= 2\gamma_u + \left(\gamma_u+\kappa\right)\left(e^{\gamma_u t}-1\right)\\
    \gamma_u &= \sqrt{\kappa^2+2u\sigma^2}\\
\end{align}
