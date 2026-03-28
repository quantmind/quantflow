# Theory

QuantFlow is built around a unified mathematical framework based on **characteristic functions** and **Lévy processes**. This section introduces the core ideas that underpin the library's stochastic process models, Fourier inversion methods, and option pricing routines.

## Characteristic Functions

The [characteristic function](./characteristic.md) of a random variable $x$ is its Fourier transform under the probability measure:

$$
\Phi_{x,u} = \mathbb{E}\left[e^{iux}\right]
$$

It is the central computational object throughout the library. Unlike the probability density function, the characteristic function is always well-defined, bounded, and closed under convolution of independent random variables. This makes it the natural tool for working with Lévy processes, where densities are often unavailable in closed form.

## Lévy Processes

A [Lévy process](./levy.md) $x_t$ has independent and stationary increments, and its characteristic function factors cleanly over time:

$$
\Phi_{x_t, u} = e^{-t\,\phi_{x_1, u}}
$$

where $\phi_{x_1,u}$ is the **characteristic exponent** at unit time, given by the Lévy-Khintchine formula.

The library extends this to **time-changed Lévy processes** $y_t = x_{\tau_t}$, where $\tau_t$ is a stochastic clock driven by an intensity process $\lambda_t$. When $\tau_t$ and $x_t$ are independent, the characteristic function of $y_t$ reduces to the Laplace transform of the integrated intensity:

$$
\Phi_{y_t, u} = \mathcal{L}_{\tau_t}\!\left(\phi_{x_1, u}\right)
$$

This structure includes the Heston stochastic volatility model and its jump extensions as special cases, where the intensity process follows a CIR (Cox-Ingersoll-Ross) dynamics.

## Fourier Inversion

Given the characteristic function, the [probability density function](./inversion.md) is recovered via inverse Fourier transform. The library implements two numerical schemes:

- **Trapezoidal / Simpson integration** (default) using the Fractional FFT (FRFT), which allows the frequency and space domains to be discretized independently.
- **Standard FFT**, available as an alternative, with the constraint that $\delta_u \cdot \delta_x = 2\pi / N$.

The FRFT is preferred in practice as it achieves higher accuracy with fewer points.

## Option Pricing

[European call options](./option_pricing.md) are priced by applying the Fourier inversion machinery to the damped call payoff. For an underlying $S_t = S_0 e^{s_t}$ with log-price process $s_t = x_t - c_t$ (where $c_t$ is the convexity correction ensuring the forward is a martingale), the call price in log-moneyness $k = \ln(K/S_0)$ is:

$$
c_k = \frac{e^{-\alpha k}}{\pi} \int_0^\infty e^{-ivk}\, \Psi(v - i\alpha)\, dv, \qquad
\Psi_u = \frac{\Phi_{s_t}(u-i)}{iu(iu+1)}
$$

The same numerical transforms used for PDF inversion are reused here, making option pricing computationally efficient across all supported models.
