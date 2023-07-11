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

# Weiner Process

In this document, we use the term Weiner process $w_t$ is a Brownian motion with stadard deviation given by the parameter $\sigma$; that is to say, the one-dimensional Weiner process is defined as:

1. $w_t$ is Lévy process
2. $d w_t = w_{t+dt}-w_t \sim N\left(0, \sigma dt\right)$ where $N$ is the normal distribution

The characteristic exponent of $w$ is
\begin{equation}
    \phi_{w, u} = \frac{\sigma^2 u^2}{2}
\end{equation}

```{code-cell} ipython3
from quantflow.sp.weiner import WeinerProcess

pr = WeinerProcess(sigma=0.5)
pr
```

```{code-cell} ipython3

```
