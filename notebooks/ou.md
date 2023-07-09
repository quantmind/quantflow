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

where $z$, with $z_0 = 0$, is a [Lévy](./levy.md) process. As $z$ is used to drive the OU process, we will call z(t) a background driving Lévy
process (BDLP) in this contex.t

+++

## Gaussian OU Process

The Gaussian Ornstein-Uhlebeck process, is a OU process where the BDLP is a Brownian motion with drift $d z_t = \kappa\theta dt + \sigma dw_t$. Substituting this into the OU SDE equation yields:

\begin{equation}
    dx_t = \kappa\left(\theta - x_t\right) dt + \sigma dw_t
\end{equation}

$\theta$ is the long-term value of ${\bf x}_t$, $\sigma$ is a volatility parameter and $w_t$ is the standard Brownian motion.

+++

## Non-gaussian OU process

Non-Gaussian OU processes offer the possibility of capturing significant distributional deviations from Gaussianity and for flexible modeling of dependence structure.

Following the seminal paper of {cite:p}`ou`, we look at a model based on this SDEs
\begin{equation}
    dx_t = -\kappa x_t dt + dz_{\kappa t}
\end{equation}

The unusual timing $dz_{\kappa t}$ is deliberately chosen so that it will turn out that whatever the value of of $\kappa$, the marginal distribution of of $x_t$ will be unchanged. Hence we separately parameterise the distribution of the volatility and the dynamic structure.

The $z_t$ has positive increments and no drift. This type of process is called a subordinator {cite:p}`bertoin`.

One of the advantages of these OU processes is that they offer a great deal of analytical tractability. For example, the integrated value of the process, useful for integrated volatility applications, is given by
\begin{align}
   \int_0^t x_s ds &= \left(1 - e^{-\kappa t}\right)x_0 + \int_0^t \left[1 - e^{-\kappa\left(t - s\right)}\right] d z_{\kappa s}\\
    &= \frac{z_{\kappa t} - x_t + x_0}{\kappa}
\end{align}

### Lévy density

It is possible to show, see {cite:p}`ou`, that given the Lévy density $w$ of $z$, in other words, the density of the Lévy measure of the Lévy-Khintchine representation of the BDLP $z_1$, than it is possible to obtain the density $u$ of $x$ via
\begin{equation}
    w_y = -u_y - y \frac{d u_y}{d y}
\end{equation}

## Gamma OU Process

The library provides an implementation of the Non-gaussian OU process in the form of a Gamma OU process, where the invariant distribution of $x_t$ is a [gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution $\Gamma\left(c, \alpha\right)$.

In this case, the BDLP is an exponential compound Poisson process with Lévy density $r \alpha e^{-\alpha x}$, in other wordes the [exponential compount Poisson](./poisson.md) process with intensity $r$ and decay $\alpha$.

```{code-cell} ipython3
from quantflow.sp.ou import GammaOU

pr = GammaOU.create(decay=10, kappa=5)
pr
```

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
