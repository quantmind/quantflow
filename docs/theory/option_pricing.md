# Option Pricing

We can use the tooling from characteristic function inversion to price European call options on an underlying $S_t = S_0 e^{s_t}$, where $S_0$ is the spot price at time 0.

## Convexity Correction

We assume an interest rate of 0, so that the forward price equals the spot price. This assumption leads to the following no-arbitrage condition

\begin{align}
s_t &= x_t - c_t \\
{\mathbb E}_0\left[e^{s_t} \right] &= {\mathbb E}_0\left[e^{x_t - c_t} \right] = e^{-c_t} {\mathbb E}_0\left[e^{x_t} \right] = e^{-c_t} e^{-\phi_{x_t, -i}} = 1
\end{align}

Therefore, $c_t$ represents the so-called convexity correction term, and it is equal to

\begin{equation}
  c_t = -\phi_{x_t, -i}
\end{equation}

The characteristic function of $s_t$ is given by

\begin{equation}
 \Phi_{s_t}\left(u\right) = \Phi_x\left(u\right) e^{-i u c_t}
\end{equation}

As you can see the convexity correction increases with time horizon. Let us take a few examples:

### Weiner Process

This is the well-known convexity correction which appears in all diffusion-driven SDEs:

\begin{equation}
    c_t = \frac{\sigma^2 t}{2}
\end{equation}

```python
from quantflow.sp.weiner import WeinerProcess
pr = WeinerProcess(sigma=0.5)
-pr.characteristic_exponent(1, complex(0,-1))
```

which is the same as

```python
pr.convexity_correction(1)
```

## Call Option

The price $C$ of a call option with strike $K$ is defined as
\begin{align}
C &= S_0 c_k \\
k &= \ln\frac{K}{S_0}\\
c_k &= {\mathbb E}\left[\left(e^{s_t} - e^k\right)1_{s_t\ge k}\right]
\end{align}

We follow Carr and Madan (1999) and write the Fourier transform of the call option as

\begin{equation}
\Psi_u = \int_{-\infty}^\infty e^{i u k} c_k dk
\end{equation}

Note that $c_k$ tends to $e^{s_t}$ as $k \to -\infty$, therefore the call price function is not square-integrable. In order to obtain integrability, we choose complex values of $u$ of the form
\begin{equation}
u = v - i \alpha
\end{equation}
The value of $\alpha$ is a numerical choice we can check later.

It is possible to obtain the analytical expression of $\Psi_u$ in terms of the characteristic function $\Phi_s$. Once we have that expression, we can use the Fourier transform tooling presented previously to calculate option prices in this way

\begin{align}
c_k &= \int_0^{\infty} e^{-iuk} \Psi\left(u\right) du \\
    &= \frac{e^{-\alpha k}}{\pi} \int_0^{\infty} e^{-ivk} \Psi\left(v-i\alpha\right) dv \\
\end{align}

The analytical expression of $\Psi_u$ is given by

\begin{equation}
\Psi_u = \frac{\Phi_{s_t}\left(u-i\right)}{iu \left(iu + 1\right)}
\end{equation}

To integrate, we use the same approach as the PDF integration.

### Choice of $\alpha$

Positive values of $\alpha$ assist the integrability of the modified call value over the
negative moneyness axis, but aggravate the same condition for the positive moneyness axis. For the modified call value to be integrable in the positive moneyness
direction, and hence for it to be square-integrable as well, a sufficient condition
is provided by $\Psi_{-i\alpha}$ being finite, which means the characteristic function $\Phi_{t,{-(\alpha+1)i}}$ is finite.

## Black Formula

Here we illustrate how to use the characteristic function integration with the classical [Weiner process](https://en.wikipedia.org/wiki/Wiener_process).

```python
from quantflow.sp.weiner import WeinerProcess
from quantflow.options.bs import black_call

ttm = 1
p = WeinerProcess(sigma=0.5)
m = p.marginal(ttm)
m.std()
```

```python
import plotly.express as px
import plotly.graph_objects as go

r = m.call_option(64, max_moneyness=1.0)
b = black_call(r.x, p.sigma, ttm)
fig = px.line(x=r.x, y=r.y, markers=True, labels=dict(x="moneyness", y="call price"))
fig.add_trace(go.Scatter(x=r.x, y=b, name="analytical", line=dict()))
fig.show()
```
