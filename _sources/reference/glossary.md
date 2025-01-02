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

# Glossary

## Characteristic Function

The characteristic function of a random variable $X$ is the Fourier transform of $f_X$, where $f_X$ is the probability density function
of $X$
\begin{equation}
 \Phi_{X,u} = {\mathbb E}\left[e^{i u X_t}\right] = \int e^{i u x} f_X\left(x\right) dx
\end{equation}

## Cumulative Distribution Function (CDF)

The cumulative distribution function (CDF), or just distribution function,
of a real-valued random variable $X$ is the function given by
\begin{equation}
    F_X(x) = P(X \leq x)
\end{equation}

## Hurst Exponent

The Hurst exponent is a measure of the long-term memory of time series. The Hurst exponent is a measure of the relative tendency of a time series either to regress strongly to the mean or to cluster in a direction.

Check this study on the [Hurst exponent with OHLC data](./applications/hurst).

## Moneyness

Moneyness is used in the context of option pricing and it is defined as

\begin{equation}
    \ln\frac{K}{F}
\end{equation}

where $K$ is the strike and $F$ is the Forward price. A positive value implies strikes above the forward, which means put options are in the money and call options are out of the money.


## Moneyness Time Adjusted

The time-adjusted moneyness is used in the context of option pricing in order to compare options with different maturities. It is defined as

\begin{equation}
    \frac{1}{\sqrt{T}}\ln\frac{K}{F}
\end{equation}

where $K$ is the strike and $F$ is the Forward price and $T$ is the time to maturity.

The key reason for dividing by the square root of time-to-maturity is related to how volatility and price movement behave over time.
The price of the underlying asset is subject to random fluctuations, if these fluctuations follow a Brownian motion than the
standard deviation of the price movement will increase with the square root of time.

## Probability Density Function (PDF)

The [probability density function](https://en.wikipedia.org/wiki/Probability_density_function)
 (PDF), or density, of a continuous random variable, is a function that describes the relative likelihood for this random variable to take on a given value. It is related to the CDF by the formula

\begin{equation}
    F_X(x) = \int_{-\infty}^x f_X(s) ds
\end{equation}
