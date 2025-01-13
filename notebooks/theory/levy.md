---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Lévy process
A Lévy process $x_t$ is a stochastic process which satisfies the following properties

* $x_0 = 0$
* **independent increments**: $x_t - x_s$ is independent of $x_u; u \le s\ \forall\ s < t$
* **stationary increments**: $x_{s+t} - x_s$ has the same distribution as $x_t - x_0$ for any $s,t > 0$

This means that the shocks to the process are independent, while the stationarity assumption specifies that the distribution of $x_{t+s} - x_s$ may change with $s$ but does not depend upon $t$.

**Remark**: The properties of stationary and independent increments implies that a Lévy process is a Markov process.
Thanks to almost sure right continuity of paths, one may show in addition that Lévy processes are also
Strong Markov processes. See ([Markov property](https://en.wikipedia.org/wiki/Markov_property)).

## Characteristic function

The independence and stationarity of the increments of the Lévy process imply that the [characteristic function](./characteristic.md) of $x_t$ has the form

\begin{equation}
 \Phi_{x_t, u} = {\mathbb E}\left[e^{i u x_t}\right] = e^{-\phi_{x_t, u}} = e^{-t \phi_{x_1,u}}
\end{equation}

where the [](characteristic-exponent) $\phi_{x_1,u}$ is given by the [Lévy–Khintchine formula](https://en.wikipedia.org/wiki/L%C3%A9vy_process).

There are several Lévy processes in the literature, including, the [Poisson process](../models/poisson.md), the compound Poisson process
and the [Brownian motion](../models/weiner.md).

+++

## Time Changed Lévy Processes

We follow the paper by Carr and Wu {cite:p}`carr_wu` to defined a continuous time changed Lévy process $y_t$ as

\begin{align}
y_t &= x_{\tau_t}\\
\tau_t &= \int_0^t \lambda_s ds
\end{align}

where $x_s$ is a Lévy process and $\lambda_s$ is a positive and integrable process which we refer to **stochastic intensity process**.
While $\tau_t$ is always continuous, $\lambda$ can exhibit jumps. Since the time-changed process is a stochastic process evaluated at a stochastic time, its characteristic function involves expectations over two sources of randomness:

\begin{equation}
 \Phi_{y_t, u} = {\mathbb E}\left[e^{i u x_{\tau_t}}\right] = {\mathbb E}\left[{\mathbb E}\left[\left.e^{i u x_s}\right|\tau_t=s\right]\right]
\end{equation}

where the inside expectation is taken on $x_{\tau_t}$ conditional on a fixed value of $\tau_t = s$ and the outside expectation is on all possible values of $\tau_t$. If the random time $\tau_t$ is independent of $x_t$, the randomness due to the Lévy process can be integrated out using the characteristic function of $x_t$:

\begin{equation}
\Phi_{y_t, u} = {\mathbb E}\left[e^{-\tau_t \phi_{x,u}}\right] = {\mathbb L}_{\tau_t}\left(\phi_{x_1,u}\right)
\end{equation}

**Remark**: Under independence, the characteristic function of a time-changed Lévy process $y_t$ is the **Laplace transform** of the cumulative intensity $\tau_t$ evaluated at the characteristic exponent of $x$.

Therefore the characteristic function of $y_t$ can be expressed in closed form if

* the characteristic exponent of the Lévy process $x_t$ is available in closed from
* the Laplace transform of $\tau_t$, the integrated intensity process, is known in closed from

## Leverage Effect

To obtain the Laplace transform  of $\tau_t$ in closed form, consider its specification in terms of the intensity process $\lambda_t$:

\begin{equation}
{\mathbb L}_{\tau_t}\left(u\right) = {\mathbb E}\left[e^{- u \int_0^t \lambda_s ds}\right]
\end{equation}

This equation is very common in the bond pricing literature if we regard $u\lambda_t$ as the instantaneous interest rate.
In the general case, the intensity process is correlated with the Lévy process of increments, this is well
known in the literature as the **leverage effect**.

Carr and Wu {cite:p}`carr_wu` solve this problem by changing the measure from an economy with leverage effect to one without it.

\begin{align}
\Phi_{y_t, u} &= {\mathbb E}\left[e^{i u y_t}\right] \\
     &= {\mathbb E}\left[e^{i u y_t + \tau_t \phi_{x_1, u} - \tau_t \phi_{x_1, u}}\right] \\
     &= {\mathbb E}\left[M_{t, u} e^{-\tau_t \phi_{x_1,u}}\right] \\
     &= {\mathbb E}^u\left[e^{-\tau_t \phi_{x_1,u}}\right] \\
     &= {\mathbb L}_{\tau_t}^u\left(\phi_{x_1,u}\right)
\end{align}

where $E[\cdot]$ and $E^u[\cdot]$ denote the expectation under probability measure $P$ and $Q^u$, respectively. The two measures are linked via
the complex-valued [Radon–Nikodym derivative](https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_theorem#Radon%E2%80%93Nikodym_derivative)

\begin{equation}
M_{t, u} = \frac{d Q^u}{d P} = \exp{\left(i u y_t + \tau_t \phi_{x_1, u}\right)} = \exp{\left(i u y_t + \phi_{x_1, u}\int_0^t \lambda_s ds\right)}
\end{equation}

## Affine definition

In order to obtain analytically tractable models we need to impose some restriction on the stochastic intensity process.
An affine intensity process takes the general form

\begin{equation}
    v_t = r_0 + r z_t
\end{equation}

where $r_0$ and $r_1$ are contants and ${\bf z}_t$ is a Markov process called the **state process**.
When the intensity process is affine, the Laplace transform takes the following form.

\begin{equation}
{\mathbb L}_{\tau_t}\left(z\right) = {\mathbb E}\left[e^{- z \tau_t}\right] = e^{-a_{u, t} - b_{u, t} z_0}
\end{equation}

where coefficients $a$ and $b$ satisfy Riccati ODEs, which can be solved numerically and, in some cases, analytically.

```{code-cell}

```
