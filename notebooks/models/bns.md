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

# BNS Model

The Barndorff-Nielson--Shephard (BNS) model is a stochastic volatility model where the variance process $\nu_t$, or better, the activity rate process, follows a [non-gaussian OU process](./ou.md). The leverage effect can be accommodated by correlating the Brownian motion $w_t$ and the BDLP $z_t$ as the following equations illustrate:

\begin{align}
    y_t &= w_{\tau_t} + \rho z_{\kappa t} \\
    d \nu_t &= -\kappa \nu_t dt + d z_{\kappa t} \\
    \tau_t &= \int_0^t \nu_s ds
\end{align}

This means that the characteristic function of $y_t$ can be represented as

\begin{align}
    \Phi_{y_t, u} & = {\mathbb E}\left[\exp{\left(i u w_{\tau_t} + i u \rho z_{\kappa t}\right)}\right] \\
    &= {\mathbb E}\left[\exp{\left(-\tau_t \phi_{w, u} + i u \rho z_{\kappa t}\right)}\right]
\end{align}

$\phi_{w, u}$ is the characteristic exponent of $w_1$. The second equivalence is a consequence of $w$ and $\tau$ being independent, as discussed in [the time-changed Lévy](./levy.md) process section.

```{code-cell} ipython3
from quantflow.sp.bns import BNS

pr = BNS.create(vol=0.5, decay=10, kappa=10, rho=-1)
pr
```

```{code-cell} ipython3
from quantflow.utils import plot
m = pr.marginal(2)
plot.plot_characteristic(m, max_frequency=10)
```

## Marginal Distribution

```{code-cell} ipython3
m.mean(), m.std()
```

```{code-cell} ipython3
plot.plot_marginal_pdf(m, 128, normal=True, analytical=False)
```

## Appendix


Carr at al {cite:p}`cgmy` show that the join characteristic function of $\tau_t$ and $z_{\kappa t}$ has a closed formula, and this is our derivation

\begin{align}
    \zeta_{a, b} &= \ln {\mathbb E} \left[\exp{\left(i a \tau_t + i b z_{\kappa t}\right)}\right] \\
    \zeta_{a, b} &= i c \nu_0 - \int_b^{b+c} \frac{\phi_{z_1, s}}{a+\kappa b - \kappa s} ds = i c \nu_0 + \lambda \left(I_{b+c} - I_{b}\right) \\
    c &= a \frac{1 - e^{-\kappa t}}{\kappa}
\end{align}


Noting that (see [non-gaussian OU process](./ou.md))

\begin{align}
i a \tau_t + i b z_{\kappa t} &= i a \epsilon_t \nu_0 + \int_0^t \left(i a \epsilon_{t-s} + i b\right) d z_{\kappa s} \\
&= i a \epsilon_t \nu_0 + \int_0^{\kappa t} \left(i a \epsilon_{t-s/\kappa} + i b\right) d z_s \\
\epsilon_t &= \frac{1 - e^{-\kappa t}}{\kappa}
\end{align}

we obtain
\begin{align}
    \zeta_{a, b} &= i a \epsilon_t \nu_0 + \ln {\mathbb E} \left[\exp{\left(\int_0^{\kappa t} \left(i a \epsilon_{t-s/\kappa} + i b\right) d z_s\right)}\right] \\
    &=  i a \epsilon_t \nu_0 - \int_0^{\kappa t} \phi_z\left(a \epsilon_{t-s/\kappa} + b\right) d s  \\
     &=  i a \epsilon_t \nu_0 - \int_L^U \frac{\phi_{z,s}}{a + \kappa b - \kappa s} d s
\end{align}

Here we use [sympy](https://www.sympy.org/en/index.html) to derive the integral in the characteristic function.

```{code-cell} ipython3
import sympy as sym
```

```{code-cell} ipython3
k =  sym.Symbol("k")
iβ = sym.Symbol("iβ")
γ = sym.Symbol("γ")
s = sym.Symbol("s")
ϕ = s/(s+iβ)/(γ-k*s)
ϕ
```

```{code-cell} ipython3
r = sym.integrate(ϕ, s)
sym.simplify(r)
```

```{code-cell} ipython3
import numpy as np
f = lambda x: x*np.log(x)
f(0.001)
```

```{code-cell} ipython3

```
