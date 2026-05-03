# Convexity Correction

When pricing derivatives we work under a risk-neutral measure and require that the
discounted asset price is a martingale. If we model the log-return directly as a
stochastic process $x_t$, that process will generally not satisfy this condition on
its own. The convexity correction is the deterministic adjustment that restores it.

## Setup

Consider an asset price of the form

\begin{equation}
S_t = S_0\, e^{s_t}
\end{equation}

where $s_t$ is the log-return. We model $s_t$ as

\begin{equation}
s_t = x_t - c_t
\end{equation}

where $x_t$ is the driving stochastic process (e.g. a Brownian motion, a Lévy process,
or a stochastic-volatility process) and $c_t$ is a deterministic function of time.

The no-arbitrage condition (with zero interest rates) requires the forward price to
equal the spot price, i.e. ${\mathbb E}[S_t] = S_0$, which is equivalent to

\begin{equation}
{\mathbb E}\!\left[e^{s_t}\right] = 1
\end{equation}

## The Correction Term

Substituting $s_t = x_t - c_t$ into the martingale condition gives

\begin{align*}
{\mathbb E}\!\left[e^{x_t - c_t}\right] &= 1 \\
e^{-c_t}\,{\mathbb E}\!\left[e^{x_t}\right] &= 1
\end{align*}

so the correction must satisfy

\begin{equation}
c_t = \log {\mathbb E}\!\left[e^{x_t}\right]
\end{equation}

In terms of the [characteristic exponent](characteristic.md) $\phi_{x_t, u}$ (defined by
$\Phi_{x_t}(u) = e^{-\phi_{x_t,u}}$), evaluating at $u = -i$ gives

\begin{equation}
{\mathbb E}\!\left[e^{x_t}\right] = \Phi_{x_t}(-i) = e^{-\phi_{x_t,-i}}
\end{equation}

therefore

\begin{equation}
c_t = -\phi_{x_t, -i}
\end{equation}

This is the log-Laplace transform (cumulant generating function) of $x_t$ evaluated
at 1. It is always real-valued and non-negative by Jensen's inequality, since
$e^{x_t}$ is convex (hence the name *convexity correction*).

## Effect on the Characteristic Function

The characteristic function of the log-return $s_t = x_t - c_t$ is

\begin{equation}
\Phi_{s_t}(u) = {\mathbb E}\!\left[e^{iu s_t}\right] = {\mathbb E}\!\left[e^{iu(x_t - c_t)}\right] = e^{-iuc_t}\,\Phi_{x_t}(u)
\end{equation}

The correction introduces a phase shift proportional to $c_t$. Two key values confirm
the martingale property:

* $\Phi_{s_t}(0) = 1$ (normalization, always true)
* $\Phi_{s_t}(-i) = e^{-c_t}\,{\mathbb E}[e^{x_t}] = e^{-c_t} e^{c_t} = 1$ (martingale condition)

## Wiener Process

For a Brownian motion $x_t = \sigma W_t$ the moment generating function is

\begin{equation}
{\mathbb E}\!\left[e^{x_t}\right] = e^{\sigma^2 t / 2}
\end{equation}

so the correction is

\begin{equation}
c_t = \frac{\sigma^2 t}{2}
\end{equation}

This is the classical term that appears in the Black-Scholes formula. It grows linearly
with time, reflecting the fact that geometric Brownian motion drifts upward on average
(the asset price is log-normally distributed with positive variance).

```python
from quantflow.sp.weiner import WeinerProcess
pr = WeinerProcess(sigma=0.5)
-pr.characteristic_exponent(1, complex(0,-1))  # c_t at t=1
```

which is the same as

```python
pr.convexity_correction(1)
```

## General Lévy Process

For any Lévy process $x_t$ with characteristic exponent $\psi$ (so that
$\phi_{x_t, u} = t\,\psi(u)$), the correction is

\begin{equation}
c_t = -t\,\psi(-i) = t\,\log {\mathbb E}\!\left[e^{x_1}\right]
\end{equation}

The correction is linear in time for all Lévy processes. For processes with jumps
(e.g. Merton jump-diffusion, variance gamma) the jump component contributes an
additional positive term on top of the diffusion $\sigma^2 t / 2$.
