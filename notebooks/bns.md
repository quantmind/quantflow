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

The Barndorff-Nielson--Shephard (BNS) model is a stochastic volatility model where the the variance process, or better, the activity rate process, follows a [non-gaussian OU process](./ou.md). The leverage effect can be accommodated by correlating the two Brownian motions as the following equations illustrate:

\begin{align}
    y_t &= w_{\tau_t} + \rho z_{\kappa t} \\
    d \nu_t &= -\kappa \nu_t dt + d z_{\kappa t} \\
    \tau_t &= \int_0^t \nu_s ds
\end{align}

This means that the characteristic function of $y_t$ can be represented as

\begin{align}
    \Phi_{y_t, u} & = {\mathbb E}\left[\exp{\left(i u w_{\tau_t} + i u z_{\kappa t}\right)}\right] \\
    &= {\mathbb E}\left[\exp{\left(-\tau_t \phi_{w, u} + i u z_{\kappa t}\right)}\right]
\end{align}

The second equivalence is a consequence of $w$ and $\tau$ being independent, as discussed in [the time-changed Lévy](./levy.md) process section.

## Characteristic Function

Carr at al {cite:p}`cgmy` show that the join characteristic function of $\tau_t$ and $z_{\kappa t}$ has a closed formula given by

\begin{align}
    \Psi_{a, b} &= {\mathbb E} \left[\exp{\left(i a \tau_t + i b z_{\kappa t}\right)}\right] \\
    &= i c \nu_0 + \int_b^{b+c} \frac{\phi_z\left(s\right)}{a+\kappa b - \kappa s} ds \\
    &= i c \nu_0 + I_{b+c} - I_{b} \\
    c &= a \frac{1 - e^{-\kappa t}}{\kappa}
\end{align}

Recall that in the case of the $\Gamma$-OU process, $z$ is an exponential compound process and its characteristic exponent is given by

\begin{equation}
    \phi_{z,u} = \frac{i u \lambda}{iu - \beta}
\end{equation}

Substituting into the integral in $\Psi_{a,b}$ one obtains the value of $I$ as

\begin{align}
I_x &= \frac{\lambda}{\kappa -i g} \ln{\left(\beta + x\right)} + \frac{\lambda + g}{\kappa\left(g+i\kappa\right)} \ln{\left(\beta g - \kappa x\right)}\\
g &= \frac{a + \kappa b}{\beta}
\end{align}



```{code-cell} ipython3

```
