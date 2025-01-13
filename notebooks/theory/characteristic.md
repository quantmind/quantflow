---
jupytext:
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

# Characteristic Function

The library makes heavy use of [characteristic function](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory))
concept and therefore, it is useful to familiarize with it.

## Definition

The characteristic function of a random variable $x$ is the Fourier (inverse) transform of ${\mathbb P}_x$, where ${\mathbb P}_x$ is the distrubution measure of $x$
\begin{equation}
 \Phi_{x,u} = {\mathbb E}\left[e^{i u x}\right] = \int e^{i u s} {\mathbb P}_x\left(ds\right)
\end{equation}

## Properties

* $\Phi_{x, 0} = 1$
* it is bounded, $\left|\Phi_{x, u}\right| \le 1$
* it is Hermitian, $\Phi_{x, -u} = \overline{\Phi_{x, u}}$
* it is continuous
* characteristic function of a symmetric random variable is real-valued and even
* moments of $x$ are given by

\begin{equation}
    {\mathbb E}\left[x^n\right] = i^{-n} \left.\frac{\Phi_{x, u}}{d u}\right|_{u=0}
\end{equation}

## Covolution

The characteristic function is a great tool for working with linear combination of random variables.

* if $x$ and $y$ are independent random variables then the characteristic function of the linear combination $a x + b y$ ($a$ and $b$ are constants) is

\begin{equation}
    \Phi_{ax+bx,u} = \Phi_{x,a u}\Phi_{y,b u}
\end{equation}

* which means, if $x$ and $y$ are independent, the characteristic function of $x+y$ is the product

\begin{equation}
    \Phi_{x+x,u} = \Phi_{x,u}\Phi_{y,u}
\end{equation}

* The characteristic function of $ax+b$ is

\begin{equation}
    \Phi_{ax+b,u} = e^{iub}\Phi_{x,au}
\end{equation}

## Inversion

There is a one-to-one correspondence between cumulative distribution functions and characteristic functions, so it is possible to find one of these functions if we know the other.

### Continuous distributions

The inversion formula for these distributions is given by

\begin{equation}
    {\mathbb P}_x\left(x=s\right) = \frac{1}{2\pi}\int_{-\infty}^\infty e^{-ius}\Phi_{s, u} du
\end{equation}

### Discrete distributions

In these distributions, the random variable $x$ takes integer values $k$. For example, the Poisson distribution is discrete.
The inversion formula for these distributions is given by

\begin{equation}
    {\mathbb P}_x\left(x=k\right) = \frac{1}{2\pi}\int_{-\pi}^\pi e^{-iuk}\Phi_{k, u} du
\end{equation}

```{code-cell}

```

(characteristic-exponent)=
## Characteristic Exponent

The characteristic exponent $\phi_{x,u}$ is defined as

\begin{equation}
 \Phi_{x,u} = e^{-\phi_{x,u}}
\end{equation}
