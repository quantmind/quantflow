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

# Glossary

## Characteristic Function

The characteristic function of a random variable $x$ is the Fourier transform of $P^x$, where $P^x$ is the distrubution measure of $x$
\begin{equation}
 \Phi_{x,u} = {\mathbb E}\left[e^{i u x_t}\right] = \int e^{i u x} P^x\left(dx\right)
\end{equation}
## Moneyness

Monenyness is used in the context of option pricing and it is defined as

\begin{equation}
    \ln\frac{K}{F}
\end{equation}

where $K$ is the strike and $F$ is the Forward price. A positive value implies strikes above the forward, which means put options are in the money and call options are out of the money.

```{code-cell} ipython3

```
