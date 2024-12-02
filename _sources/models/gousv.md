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

# Gaussian OU Vol Model


\begin{align}
    d x_t &= d w_t \\
    d \eta_t &= \kappa\left(\theta - \eta_t\right) dt + \sigma d b_t \\
    \tau_t &= \int_0^t \eta_s^2 ds \\
    {\mathbb E}\left[d w_t d b_t\right] &= \rho dt
\end{align}

This means that the characteristic function of $y_t=x_{\tau_t}$ can be represented as

\begin{align}
    \Phi_{y_t, u} & = {\mathbb E}\left[e^{i u y_t}\right] = {\mathbb L}_{\tau_t}^u\left(\frac{u^2}{2}\right) \\
     &= e^{-a_{t,u} - b_{t,u} \nu_0}
\end{align}

```{code-cell} ipython3

```

## Characteristic Function

\begin{align}
    a_t &= \left(\theta - \frac{\sigma^2}{2\kappa^2}\right)\left(b_t -t\right) - \frac{\sigma^2}{4\kappa}b_t^2 \\
    b_t &= \frac{1 - e^{-\kappa t}}{\kappa} \\
\end{align}

```{code-cell} ipython3

```

```{code-cell} ipython3

```
