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

# Poisson Processes

In this section, we look at the family of pure jump processes which are Lévy provcesses.
The most common process is the Poisson process.

## Poisson Process

```{code-cell} ipython3
from quantflow.sp.poisson import PoissonProcess
pr = PoissonProcess(intensity=2)
pr
```

### Marginal

```{code-cell} ipython3
import numpy as np
from quantflow.utils import plot

m = pr.marginal(1)
plot.plot_marginal_pdf(m, np.arange(10), analytical="markers")
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

The characteristic exponent is given by

\begin{align}
  e = \int_0^\infty \left(e^{iuy} - 1\right) f(y) dy
\end{align}


The library includes the Exponential Poisson Process, a compound Poisson process where the jump sizes are sampled from an exponential distribution.

```{code-cell} ipython3
from quantflow.sp.poisson import ExponentialPoissonProcess

pr = ExponentialPoissonProcess(rate=1, decay=10)
pr
```

```{code-cell} ipython3
pr.sample(10, time_horizon=10, time_steps=1000).plot().update_traces(line_width=1)
```

## Doubly Stochastic Poisson Process


The aim is to identify a stochastic process for simulating the goal arrival which fulfills the following properties

* Capture overdispersion
* Analytically tractable
* Capture the inherent randomness of the goal intensity
* Intuitive

Before we dive into the details of the DSP process, lets take a quick tour of what Lévy processes are, how a time chage can open the doors to a vast array of models and why they are important in the context of DSP.of DSP.

+++

## DSP process

Here we are interested in a special Lévy process, a Poisson process $p_t$ with intensity equal to 1. The characteristic exponent of this process is given by

\begin{equation}
\phi_{p,u} = e^{iu} - 1
\end{equation}

The DSP is defined as
\begin{equation}
 N_t = p_{\tau_t}
\end{equation}

where $\tau_t$ is the **cumulative intensity**, or the **hazard process**, for the intensity process $\lambda_t$.
The Characteristic function of $N_t$ can therefore be written as

\begin{equation}
    \Phi_{N_t, u} = {\mathbb E}\left[e^{-\tau_t \left(e^{iu}-1\right)}\right]
\end{equation}


The doubly stochastic Poisson process (DSP process) with intensity process $\lambda_t$ is a point proces $y_t = p_{\Lambda_t}$
satisfying the following expression for the conditional distribution of the n-th jump

\begin{equation}
{\mathbb P}\left(\tau_n > T\right) = {\mathbb E}_t\left[e^{-\Lambda_{t,T}} \sum_{j=0}^{n-1}\frac{1}{j!} \Lambda_{t, T}^j\right]
\end{equation}

The intensity function of a DSPP is given by:

\begin{equation}
{\mathbb P}\left(N_T - N_t = n\right) = {\mathbb E}_t\left[e^{-\Lambda_{t,T}} \frac{\Lambda_{t, T}^n}{n!}\right] = \frac{1}{n!}
\end{equation}

```{code-cell} ipython3

```
