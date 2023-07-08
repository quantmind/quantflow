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

## Poisson Process

```{code-cell} ipython3
from quantflow.sp.poisson import PoissonProcess
pr = PoissonProcess(rate=2)
pr
```

```{code-cell} ipython3
p = pr.sample(10, time_horizon=1, time_steps=1000)
p.plot()
```

## Compound Poisson Process

The compound poisson process is a jump process, where the arrival of jumps follows the same dynamic as the Poisson process but the size of jumps is no longer constant and equal to 1, instead it follows a given distribution.

The library includes the Exponential Poisson Process, a compound Poisson process where the jump sizes are sampled from an exponential distribution.

```{code-cell} ipython3
from quantflow.sp.poisson import ExponentialPoissonProcess

pr = ExponentialPoissonProcess(rate=1, decay=10)
pr
```

```{code-cell} ipython3
pr.sample(10, time_horizon=1, time_steps=1000).plot()
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
