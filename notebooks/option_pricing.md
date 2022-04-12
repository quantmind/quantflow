---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Option Pricing


We can use the tooling from characteristic function inversion to price european call options on an underlying $S_t = S_0 e^{s_t}$, where $S_0$ is the spot price at time 0.

## Convexity Correction

We assume interest rate 0, so that the forward price is equal the spot price. This assumtion leads to the following no arbitrage condition

\begin{align}
s_t &= x_t - c \\
{\mathbb E}_0\left[e^{s_t} \right] &= {\mathbb E}_0\left[e^{x_t - c} \right] = e^{-c} {\mathbb E}_0\left[e^{x_t} \right] = e^{-c} \Phi_x\left(-i\right) = 1
\end{align}

Therefore, c represents the so called convextity correction term and it is equal to

\begin{equation}
  c = \ln{\Phi_x\left(-i\right)}
\end{equation}

The characteristic function of $s_t$ is given by

\begin{equation}
 \Phi_{s_t}\left(u\right) = \Phi_x\left(u\right) e^{-i u c}
\end{equation}

## Call option

A call option is defined as
\begin{align}
C &= S_0 c_k \\
k &= \ln\frac{K}{S_0}\\ 
c_k &= {\mathbb E}\left[\left(e^{s_t} - e^k\right)1_{s_t\ge k}\right]
\end{align}


We follow {cite:p}`carr_madan` and write the Fourier transform of the the call option as

\begin{equation}
\Psi_u = \int_{-\infty}^\infty e^{i u k} c_k dk
\end{equation}

Note that $c_k$ tends to $e^x_t$ as $k \to -\infty$, therefore the call price function is not square-integrable. In order to obtain integrability, we choose complex values of $u$ of the form
\begin{equation}
u = v - i \alpha
\end{equation}
The value of $\alpha$ is a numerical choice we can check later.

It is possible to obtain the analytical expression of $\Psi_k$ in terms of the characteristic function $\Phi_x$. Once we have that expression we can use the Fourier transform tooling presented peviously to calculate option prices.

\begin{align}
c_k &= \int_0^{\infty} e^{-iuk} \Psi\left(u\right) du \\
    &= \frac{e^{-\alpha k}}{\pi} \int_0^{\infty} e^{-ivk} \Psi\left(v-i\alpha\right) dv \\
\end{align}

The analytical expression of $\phi_k$ is given by

\begin{equation}
\Psi\left(u\right) = \frac{\Phi_{s_t}\left(u-i\right)}{iu \left(iu + 1\right)}
\end{equation}

To integrate we use the same approach as the PDF integration.

```{code-cell} ipython3
from quantflow.sp.weiner import Weiner
ttm=1
p = Weiner(0.5)
m = p.marginal(ttm)
m.std()
```

```{code-cell} ipython3
import plotly.express as px
import plotly.graph_objects as go
from quantflow.utils.bs import black_call
N, M = 128, 10
dx = 10/N
r = m.call_option(N, M, dx, alpha=0.2)
b = black_call(r["x"], p.sigma.value, ttm)
fig = px.line(r, x="x", y="y", markers=True, labels=dict(x="moneyness", y="call price"))
fig.add_trace(go.Scatter(x=r["x"], y=b, name="analytical", line=dict()))
fig.show()
```

```{code-cell} ipython3

```
