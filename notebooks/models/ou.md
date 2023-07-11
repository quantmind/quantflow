---
jupytext:
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

# OU Process

The general definition of an Ornstein-Uhlebeck (OU) process is as the solution of an SDE of the form.

\begin{equation}
    d x_t = -\kappa x_t dt + d z_t
\end{equation}

where $z$, with $z_0 = 0$, is a [Lévy](./levy.md) process. As $z$ drives the OU process, we will call z(t) a background driving Lévy process (**BDLP**) in this context.

The process can be integrated to the formula

\begin{equation}
    x_t = x_0 e^{-\kappa t} + \int_0^t e^{-\kappa\left(t-s\right)} d z_{\kappa s}
\end{equation}

The characteristic exponent of this process is given by

\begin{equation}
\end{equation}

+++

## Gaussian OU Process

The Gaussian Ornstein-Uhlebeck process, is a OU process where the BDLP is a Brownian motion with drift $d z_t = \kappa\theta dt + \sigma dw_t$. Substituting this into the OU SDE equation yields:

\begin{equation}
    dx_t = \kappa\left(\theta - x_t\right) dt + \sigma dw_{\kappa t}
\end{equation}

$\theta$ is the long-term value of ${\bf x}_t$, $\sigma$ is a volatility parameter and $w_t$ is the standard Brownian motion.

+++

## Non-gaussian OU process

Non-Gaussian OU processes offer the possibility of capturing significant distributional deviations from Gaussianity and for flexible modeling of dependence structure.

Following the seminal paper of {cite:p}`ou`, we look at a model based on this SDEs
\begin{equation}
    dx_t = -\kappa x_t dt + dz_{\kappa t}
\end{equation}

The unusual timing $dz_{\kappa t}$ is deliberately chosen so that it will turn out that whatever the value of of $\kappa$, the marginal distribution of of $x_t$ will be unchanged. Hence we separately parameterize the distribution of the volatility and the dynamic structure.

The $z_t$ has positive increments and no drift. This type of process is called a subordinator {cite:p}`bertoin`.

### Integration

When the subordinator is a Compound Poisson process, then the integration takes the form

\begin{equation}
    x_t = x_0 e^{-\kappa t} + \sum_{n=0}^{N_{\kappa t}} e^{-\kappa t-m_n} j_n
\end{equation}

where $m_n$ are the jump times of the Poisson process $N_{\kappa t} and $j_n$ are the jump sizes drawn from the jump distribution.

### Integrated Intensity

One of the advantages of these OU processes is that they offer a great deal of analytical tractability. For example, the integrated value of the process, which can be used as a time change for [Lévy processes](./levy.md), is given by

\begin{align}
   \int_0^t x_s ds &= \epsilon_t x_0 + \int_0^t \epsilon_{t-s} d z_{\kappa s} = \frac{z_{\kappa t} - x_t + x_0}{\kappa}\\
   \epsilon_t &= \frac{1 - e^{-\kappa t}}{\kappa}
\end{align}

### Lévy density

It is possible to show, see {cite:p}`ou`, that given the Lévy density $w$ of $z$, in other words, the density of the Lévy measure of the Lévy-Khintchine representation of the BDLP $z_1$, than it is possible to obtain the density $u$ of $x$ via
\begin{equation}
    w_y = -u_y - y \frac{d u_y}{d y}
\end{equation}

## Gamma OU Process

The library provides an implementation of the non-gaussian OU process in the form of a Gamma OU process, where the invariant distribution of $x_t$ is a [gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution $\Gamma\left(\lambda, \beta\right)$.

In this case, the BDLP is an exponential compound Poisson process with Lévy density $\lambda\beta e^{-\beta x}$, in other words, the [exponential compound Poisson](./poisson.md) process with intensity $\lambda$ and decay $\beta$.

```{code-cell} ipython3
from quantflow.sp.ou import GammaOU

pr = GammaOU.create(decay=10, kappa=5)
pr
```

### Characteristic Function

The charatecristic exponent of the $\Gamma$-OU process is given by, see {cite:p}`gamma-ou`)

\begin{equation}
    \phi_{u, t} = -x_{0} i u e^{-\kappa t} - \lambda\ln\left(\frac{\beta-iue^{-\kappa t}}{\beta -iu}\right)
\end{equation}

```{code-cell} ipython3
pr.marginal(1).mean(), pr.marginal(1).std()
```

```{code-cell} ipython3
import numpy as np
from quantflow.utils import plot

m = pr.marginal(5)
plot.plot_marginal_pdf(m, 0.05*np.arange(100))
```

```{code-cell} ipython3
from quantflow.utils.plot import plot_characteristic
plot_characteristic(m)
```

### Sampling Gamma OU

```{code-cell} ipython3
from quantflow.sp.ou import GammaOU
pr = GammaOU.create(decay=10, kappa=5)

pr.sample(50, time_horizon=1, time_steps=1000).plot().update_traces(line_width=0.5)
```

### MC testing

Test the simulated meand and stadard deviation against the values from the invariant gamma distribution.

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
