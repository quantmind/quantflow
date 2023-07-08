---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
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

# Lévy process
A Lévy process $x_t$ is a stochastic process which satisfies the following properties

* **independent increments**: $x_t - x_s$ is independent of $x_u; u \le s\ \forall\ s < t$
* **stationary increments**: $x_{s+t} - x_s$ has the same distribution as $x_t - x_0$ for any $s,t > 0$

This means that the shocks to the process are independent, while the stationarity assumption specifies that the distribution of $x_{t+s} - x_s$ may change with $s$ but does not depend upon $t$.

**Remark**: The properties of stationary and independent increments implies that a Lévy process is a Markov process.
Thanks to almost sure right continuity of paths, one may show in addition that Lévy processes are also
Strong Markov processes. See ([Markov property](https://en.wikipedia.org/wiki/Markov_property)).

## Characteristic function

The independence and stationarity of the increments of the Lévy process implies that the [characteristic function](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)) of $x_t$ has the form

\begin{equation}
 \Phi_{x_t, u} = {\mathbb E}\left[e^{i u x_t}\right] = e^{-t \phi_{x,u}}
\end{equation}

where the **characteristic exponent** $\phi_{x,u}$ is given by the [Lévy–Khintchine formula](https://en.wikipedia.org/wiki/L%C3%A9vy_process).

There are several Levy process in the literature including, importantly, the **Poison** process, the compount Poisson process and the Brownian motion.

+++

## Time Changed Lévy Processes

We follow the paper by Carr and Wu {cite:p}`carr_wu` to defined a continuous time changed Lévy process $y_t$ as

\begin{align}
y_t &= x_{\tau_t}\\
\tau_t &= \int_0^t \lambda_s ds
\end{align}

where $x_s$ is a Lévy process and $\lambda_s$ is a positive and integrable process which we refer to **stochastic intensity process**.
Note that while $\tau_t$ is always continuous, $\lambda$ can exhibit jumps. Since the time-changed process is a stochastic process evaluated at a stochastic time, its characteristic function involves expectations over two sources of randomness:

\begin{equation}
 \Phi_{y_t, u} = {\mathbb E}\left[e^{i u x_{\tau_t}}\right] = {\mathbb E}\left[{\mathbb E}\left[\left.e^{i u x_s}\right|\tau_t=s\right]\right]
\end{equation}

where the inside expectation is taken on $x_{\tau_t}$ conditional on a fixed value of $\tau_t = s$ and the outside expectation is on all possible values of $\tau_t$. If the random time $\tau_t$ is independent of $x_t$, the randomness due to the Lévy process can be integrated out using the characteristic function of $x_t$:

\begin{equation}
\Phi_{y_t, u} = {\mathbb E}\left[e^{-\tau_t \phi_{x,u}}\right] = {\mathbb L}_{\tau_t}\left(\phi_{x,u}\right)
\end{equation}

**Remark**: Under independence, the characteristic function of a time-changed Lévy process $y_t$ is the **Laplace transform** of the cumulative intensity $\tau_t$ evaluated at the characteristic exponent of $x$.

### Affine definition

In the general case the stochastic time is correlated with increments, to obtain the Laplace transform in closed form, one consider its specification in terms of the intensity prcess $\lambda_t$:

\begin{equation}
{\mathbb L}_{\tau_t}\left(u\right) = {\mathbb E}\left[e^{- u \int_0^t \lambda_s ds}\right]
\end{equation}

In order to obtain analytically tractable models we need to impose some restriction on the stochastic intensity process.
An affine intensity process takes the general form

\begin{equation}
    v_t = r_0 + r z_t
\end{equation}

where $r_0$ and $r_1$ are contants and ${\bf z}_t$ is a Markov process called the **state process**.
When the intensity process is affine, the log characteristic function takes the following form

\begin{equation}
{\mathbb L}_{\tau_t}\left(z\right) = {\mathbb E}\left[e^{- z \tau_t}\right] = e^{-a_{u, t} - b_{u, t} z_0}
\end{equation}

+++

## Affine definition

An affine intensity process takes the general form

\begin{equation}
    \lambda_t = r_0 + {\bf r}\cdot{\bf x}_t
\end{equation}

where $r_0 \in \Re$ and ${\bf r} \in \Re^d$ are contants and ${\bf x}_t$ is a $d$-dimensional stochastic process called the **state process**.

When the intensity process is affine, the log characteristic function takes the following form

\begin{equation}
\psi_{t, T, u} = e^{a_{u, T-t} + b_{u, T-t} \lambda_t}
\end{equation}

where coefficients $a$ and $b$ satisfy Riccati ODEs which can be solved numerically and in some cases analytically.

+++

## References

* [Time-changed Lévy processes and option pricing](https://engineering.nyu.edu/sites/default/files/2019-03/Carr-time-changed-levy-processes-option-pricing.pdf)
* [Doubly Stochastic Poisson Processes
with Affine Intensities](https://editorialexpress.com/cgi-bin/conference/download.cgi?db_name=sbe35&paper_id=179)
* [Closed-form formulas for the distribution of the jumps of
doubly-stochastic Poisson processes](https://arxiv.org/pdf/1701.00717.pdf)
* [On the characteristic functional of a doubly stochastic
Poisson process](http://hera.ugr.es/doi/16516588.pdf)
* [Time Change](http://www.stats.ox.ac.uk/~winkel/winkel15.pdf)
