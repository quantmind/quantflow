# Glossary

## Characteristic Function

The [characteristic function](../theory/characteristic) of a random variable $x$ is the Fourier transform of ${\mathbb P}_x$,
where ${\mathbb P}_x$ is the distrubution measure of $x$.

\begin{equation}
 \Phi_{x,u} = {\mathbb E}\left[e^{i u x}\right] = \int e^{i u s} {\mathbb P}_x\left(d s\right)
\end{equation}

If $x$ is a continuous random variable, than the characteristic function is the Fourier transform of the PDF $f_x$.

\begin{equation}
 \Phi_{x,u} = {\mathbb E}\left[e^{i u x}\right] = \int e^{i u s} f_x\left(s\right) ds
\end{equation}


## Characteristic Exponent

The characteristic exponent $\phi_{x,u}$ is defined from the
[characteristic function](#characteristic-function) $\Phi_{x,u}$ by

\begin{equation}
    \Phi_{x,u} = e^{-\phi_{x,u}}
\end{equation}

The library implements the [characteristic_exponent][quantflow.sp.base.StochasticProcess1D.characteristic_exponent] for several stochastic processes,
including Brownian motion, Poisson and compound Poisson processes, the CIR square-root
diffusion, Ornstein-Uhlenbeck processes, Heston and Double Heston stochastic volatility
models, jump-diffusion models, and the Barndorff-Nielsen-Shephard (BNS) model.
Having an analytic form of the characteristic exponent for these processes
enables efficient option pricing via Fourier inversion methods such as the
[Lewis (2001)](bibliography.md#lewis) and [Carr-Madan (1999)](bibliography.md#carr_madan) approaches.

## Cumulative Distribution Function (CDF)

The cumulative distribution function (CDF), or just distribution function,
of a real-valued random variable $x$ is the function given by
\begin{equation}
    F_x(s) = {\mathbb P}_x(x \leq s)
\end{equation}

where ${\mathbb P}_x$ is the distrubution measure of $x$.

## Feller Condition

The Feller condition is a parameter constraint on a square-root diffusion process
(such as [CIR][quantflow.sp.cir.CIR]) that ensures the process remains strictly
positive. For a process of the form

\begin{equation}
dx_t = \kappa(\theta - x_t)\,dt + \sigma\sqrt{x_t}\,dw_t
\end{equation}

the condition is

\begin{equation}
2\kappa\theta \geq \sigma^2
\end{equation}

where $\kappa$ is the mean reversion speed, $\theta$ is the long-run mean, and $\sigma$
is the diffusion coefficient. When the condition holds, the origin is an inaccessible
boundary, so $x_t > 0$ for all $t > 0$ almost surely.

In the [Heston model][quantflow.sp.heston.Heston] the variance process $v_t$ is a CIR
process, so the same condition applies with $\sigma$ being the vol of vol. The
[CIR.is_positive][quantflow.sp.cir.CIR.is_positive] property checks whether the
condition holds. The
[HestonCalibration][quantflow.options.heston_calibration.HestonCalibration] class provides a
`feller_enforce` flag (default `True`) that imposes this as a hard inequality constraint
during optimisation.

## Hurst Exponent

The Hurst exponent is a measure of the long-term memory of time series. The Hurst exponent is a measure of the relative tendency of a time series either to regress strongly to the mean or to cluster in a direction.

Check this study on the [Hurst exponent with OHLC data](../applications/hurst).

## Log-Strike

Log-strike, or log strike/forward ratio, is used in the context of option pricing and it is defined as

\begin{equation}
    k = \ln\frac{K}{F}
\end{equation}

where $K$ is the strike and $F$ is the Forward price. A positive value implies strikes above the forward, which means put options are in the money (ITM) and call options are out of the money (OTM).
The log-strike is used as input for all Black-Scholes type formulas.

## Moneyness

Moneyness is used in the context of option pricing in order to compare options with different maturities. It is defined as

\begin{equation}
    m = \frac{1}{\sqrt{\tau}}\ln{\frac{K}{F}} = \frac{k}{\sqrt{\tau}}
\end{equation}

where $K$ is the strike, $F$ is the Forward price, and $\tau$ is the time to maturity. It is used to compare options with different maturities by scaling the [log-strike](#log-strike) by the square root of time to maturity. This is because the price of the underlying asset is subject to random fluctuations, if these fluctuations follow a Brownian motion than the standard deviation of the price movement will increase with the square root of time.


## Moneyness Vol Adjusted

The vol-adjusted moneyness is used in the context of option pricing in order to compare options with different maturities and different levels of volatility. It is defined as

\begin{equation}
    m_\sigma = \frac{1}{\sigma\sqrt{\tau}}\ln\frac{K}{F}
\end{equation}

where $K$ is the strike, $F$ is the Forward price, $\tau$ is the time to maturity and $\sigma$ is the implied Black volatility.

## Parseval's Theorem

Parseval's theorem states that for two square-integrable functions $f$ and $g$ with Fourier transforms $\hat{f}$ and $\hat{g}$

\begin{equation}
\int_{-\infty}^\infty f(s)\, g(s)\, ds = \frac{1}{2\pi} \int_{-\infty}^\infty \hat{f}(u)\, \overline{\hat{g}(u)}\, du
\end{equation}

where $\overline{\hat{g}(u)}$ denotes the complex conjugate of $\hat{g}(u)$.

If $g$ is real-valued, then

\begin{equation}
\overline{\hat{g}(u)} = \hat{g}(-u)
\end{equation}

It is used in the derivation of the [Lewis option pricing formula](theory/option_pricing.md#lewis-formula).

## Probability Density Function (PDF)

The [probability density function](https://en.wikipedia.org/wiki/Probability_density_function)
 (PDF), or density, of a continuous random variable, is a function that describes the relative likelihood for this random variable to take on a given value. It is related to the CDF by the formula

\begin{equation}
    F_x(x) = \int_{-\infty}^x f_x(s) ds
\end{equation}

## Put-Call Parity

Put-call parity is a no-arbitrage relationship between the prices of European call
and put options with the same strike $K$ and maturity. Denoting forward-space prices
$c = C/F$ and $p = P/F$ (see [Black Pricing](api/options/black.md)), the relationship
reads:

\begin{equation}
    c - p = 1 - \frac{K}{F} = 1 - e^k
\end{equation}

where $k$ is the [log-strike](#log-strike).
In quoting currency terms, multiplying through by $F$:

\begin{equation}
    C - P = F - K
\end{equation}

## Time To Maturity (TTM)

Time to maturity is the time remaining until an option or forward contract expires,
expressed in years. It is calculated using a day count convention applied to the
interval between the reference date and the expiry date. For a reference date $t_0$
and expiry date $T$:

$$\tau = \text{dcf}(t_0, T)$$

where $\text{dcf}$ is the day count fraction function (Act/Act by default in quantflow).
TTM is denoted $\tau$ throughout the pricing and calibration formulas.
