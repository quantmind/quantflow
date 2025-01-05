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

# Poisson Processes

In this section, we look at the family of pure jump processes which are Lévy processes.
The most common process is the Poisson process.

## Poisson Process

The Poisson Process $N_t$ with intensity parameter $\lambda > 0$ is a Lévy process with values in $N$ such that each $N_t$ has a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) with parameter $\lambda t$, that is

\begin{equation}
    P\left(N_t=n\right) = \frac{\left(\lambda t\right)^n}{n!}e^{-\lambda t}
\end{equation}

The characteristic exponent is given by

\begin{equation}
\phi_{N_t, u} = t \lambda \left(1 - e^{iu}\right)
\end{equation}

```{code-cell} ipython3
from quantflow.sp.poisson import PoissonProcess
pr = PoissonProcess(intensity=1)
pr
```

```{code-cell} ipython3
m = pr.marginal(0.1)
import numpy as np
cdf = m.cdf(np.arange(5))
cdf
```

```{code-cell} ipython3
n = 128*8
m.cdf_from_characteristic(5, frequency_n=n).y
```

```{code-cell} ipython3
cdf1 = m.cdf_from_characteristic(5, frequency_n=n).y
cdf2 = m.cdf_from_characteristic(5, frequency_n=n, simpson_rule=False).y
10000*np.max(np.abs(cdf-cdf1)), 10000*np.max(np.abs(cdf-cdf2))
```

### Marginal

```{code-cell} ipython3
import numpy as np
from quantflow.utils import plot

m = pr.marginal(1)
plot.plot_marginal_pdf(m, frequency_n=128*8)
```

```{code-cell} ipython3
from quantflow.utils.plot import plot_characteristic
plot_characteristic(m)
```

### Sampling Poisson

```{code-cell} ipython3
p = pr.sample(10, time_horizon=10, time_steps=1000)
p.plot().update_traces(line_width=1)
```

## Compound Poisson Process

The compound poisson process is a jump process, where the arrival of jumps $N_t$ follows the same dynamic as the Poisson process but the size of jumps is no longer constant and equal to 1, instead, they are i.i.d. random variables independent from $N_t$.

\begin{align}
  x_t = \sum_{k=1}^{N_t} j_t
\end{align}

The characteristic exponent of a compound Poisson process is given by

\begin{align}
  \phi_{x_t,u} = t\int_0^\infty \left(e^{iuy} - 1\right) f(y) dy = t\lambda \left(1 - \Phi_{j,u}\right)
\end{align}

where $\Phi_{j,u}$ is the characteristic function of the jump distribution.

As long as we have a closed-form solution for the characteristic function of the jump distribution, then we have a closed-form solution for the characteristic exponent of the compound Poisson process.

The mean and variance of the compund Poisson is given by
\begin{align}
    {\mathbb E}\left[x_t\right] &= \lambda t {\mathbb E}\left[j\right]\\
    {\mathbb Var}\left[x_t^2\right] &= \lambda t \left({\mathbb Var}\left[j\right] + {\mathbb E}\left[j\right]^2\right)
\end{align}

## Exponential Compound Poisson Process

The Exponential Poisson Process is a compound Poisson process where the jump sizes are sampled from an exponential distribution. To create an Exponential Compound Poisson process we simply pass the `Exponential` distribution as the jump distribution.

```{code-cell} ipython3
from quantflow.sp.poisson import CompoundPoissonProcess
from quantflow.utils.distributions import Exponential

pr = CompoundPoissonProcess(intensity=1, jumps=Exponential(decay=1))
pr
```

```{code-cell} ipython3
from quantflow.utils.plot import plot_characteristic
m = pr.marginal(1)
plot_characteristic(m)
```

```{code-cell} ipython3
m.mean(), m.mean_from_characteristic()
```

```{code-cell} ipython3
m.variance(), m.variance_from_characteristic()
```

```{code-cell} ipython3
pr.sample(10, time_horizon=10, time_steps=1000).plot().update_traces(line_width=1)
```

### MC simulations

Here we test the simulated mean and standard deviation against the analytical values.

```{code-cell} ipython3
import pandas as pd
from quantflow.utils import plot

paths = pr.sample(100, time_horizon=10, time_steps=1000)
mean = dict(mean=pr.marginal(paths.time).mean(), simulated=paths.mean())
df = pd.DataFrame(mean, index=paths.time)
plot.plot_lines(df)
```

```{code-cell} ipython3
std = dict(std=pr.marginal(paths.time).std(), simulated=paths.std())
df = pd.DataFrame(std, index=paths.time)
plot.plot_lines(df)
```

## Normal Compound Poisson

A compound Poisson process with a normal jump distribution

```{code-cell} ipython3
from quantflow.utils.distributions import Normal
from quantflow.sp.poisson import CompoundPoissonProcess
pr = CompoundPoissonProcess(intensity=10, jumps=Normal(mu=0.01, sigma=0.1))
pr
```

```{code-cell} ipython3
m = pr.marginal(1)
m.mean(), m.std()
```

```{code-cell} ipython3
m.mean_from_characteristic(), m.std_from_characteristic()
```

## Doubly Stochastic Poisson Process

The aim is to identify a stochastic process for simulating arrivals which fulfills the following properties

* Capture overdispersion
* Analytically tractable
* Capture the inherent randomness of the Poisson intensity
* Intuitive

The DSP process presented in {cite:p}`dspp` has an intensity process which belongs to a class of affine diffusion and it can treated analytically.

Additional links

* [Doubly Stochastic Poisson Processes
with Affine Intensities](https://editorialexpress.com/cgi-bin/conference/download.cgi?db_name=sbe35&paper_id=179)
* [Closed-form formulas for the distribution of the jumps of
doubly-stochastic Poisson processes](https://arxiv.org/pdf/1701.00717.pdf)
* [On the characteristic functional of a doubly stochastic
Poisson process](http://hera.ugr.es/doi/16516588.pdf)
* [Time Change](http://www.stats.ox.ac.uk/~winkel/winkel15.pdf)

+++

### DSP process

The DSP is defined as a time-changed Poisson process
\begin{equation}
 D_t = N_{\tau_t}
\end{equation}

where $\tau_t$ is the **cumulative intensity**, or the **hazard process**, for the intensity process $\lambda_t$.
The Characteristic function of $D_t$ can therefore be written as

\begin{equation}
    \Phi_{D_t, u} = {\mathbb E}\left[e^{-\tau_t \left(e^{iu}-1\right)}\right]
\end{equation}


The doubly stochastic Poisson process (DSP process) with intensity process $\lambda_t$ is a point process $y_t = p_{\Lambda_t}$
satisfying the following expression for the conditional distribution of the n-th jump

\begin{equation}
{\mathbb P}\left(\tau_n > T\right) = {\mathbb E}_t\left[e^{-\Lambda_{t,T}} \sum_{j=0}^{n-1}\frac{1}{j!} \Lambda_{t, T}^j\right]
\end{equation}

The intensity function of a DSPP is given by:

\begin{equation}
{\mathbb P}\left(N_T - N_t = n\right) = {\mathbb E}_t\left[e^{-\Lambda_{t,T}} \frac{\Lambda_{t, T}^n}{n!}\right] = \frac{1}{n!}
\end{equation}

```{code-cell} ipython3
from quantflow.sp.dsp import DSP, PoissonProcess, CIR
pr = DSP(intensity=CIR(sigma=2, kappa=1), poisson=PoissonProcess(intensity=2))
pr2 = DSP(intensity=CIR(rate=2, sigma=4, kappa=2, theta=2), poisson=PoissonProcess(intensity=1))
pr, pr2
```

```{code-cell} ipython3
import numpy as np
from quantflow.utils import plot
import plotly.graph_objects as go

n=16
m = pr.marginal(1)
pdf = m.pdf_from_characteristic(n)
fig = plot.plot_marginal_pdf(m, n, analytical=False, label=f"rate={pr.intensity.rate}")
plot.plot_marginal_pdf(pr2.marginal(1), n, analytical=False, fig=fig, marker_color="yellow", label=f"rate={pr2.intensity.rate}")
fig.add_trace(go.Scatter(x=pdf.x, y=pr.poisson.marginal(1).pdf(pdf.x), name="Poisson", mode="markers", marker_color="blue"))
```

```{code-cell} ipython3
pr.marginal(1).mean(), pr.marginal(1).variance()
```

```{code-cell} ipython3
pr2.marginal(1).mean(), pr2.marginal(1).variance()
```

```{code-cell} ipython3
from quantflow.utils.plot import plot_characteristic
m = pr.marginal(2)
plot_characteristic(m)
```

```{code-cell} ipython3
pr.sample(10, time_horizon=10, time_steps=1000).plot().update_traces(line_width=1)
```

```{code-cell} ipython3
m.characteristic(2)
```

```{code-cell} ipython3
m.characteristic(-2).conj()
```
