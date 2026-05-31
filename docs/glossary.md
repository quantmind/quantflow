# Glossary

## Characteristic Function

The [characteristic function](theory/characteristic.md) of a random variable $x$ is the Fourier transform of ${\mathbb P}_x$,
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

## Discount Factor

The discount factor $D\left(\tau\right)$ is the present value of one unit of currency paid at [time to maturity](#time-to-maturity-ttm) $\tau$.
It is equal to the price of a zero-coupon bond maturing at $\tau$: a contract that pays exactly 1 at maturity with no intermediate cashflows.

\begin{equation}
    D\left(\tau\right) = e^{-r \tau}
\end{equation}

where $r$ is the continuously compounded risk-free rate.

Under a zero interest rate assumption, $D\left(\tau\right) = 1\ \ \ \forall\ \tau$.

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
[HestonCalibration][quantflow.options.calibration.heston.HestonCalibration] class provides a
`feller_enforce` flag (default `True`) that imposes this as a hard inequality constraint
during optimisation.

## Forwards

The forward price $F$ of an asset at maturity $\tau$ is the price agreed upon today for delivery of the asset at time $\tau$. It is given by the ratio of two discount factors: one for the asset $D_a(\tau)$ and one for the quote $D_q(\tau)$.

\begin{equation}
F(\tau) = S \cdot \frac{D_a(\tau)}{D_q(\tau)}
\end{equation}

See [Forwards and Discount Factors](theory/forwards.md) for more details.

## Forward Space

Forward space is the unit-free convention in which option prices are
normalised by the forward price.

For a call $C$ and put $P$ with strike $K$, maturity $T$, and forward $F$,
the forward-space prices are

\begin{equation}
    c = \frac{C}{F}, \qquad p = \frac{P}{F}
\end{equation}

Forward-space prices are dimensionless and depend only on the
[log-strike](#log-strike) $k = \log(K/F)$, the implied volatility,
and the time to maturity. They are the natural output of Fourier-based
pricers and of [Black pricing](api/options/black.md).

The conversion to quote-currency prices is a single multiplication by $F$:

\begin{equation}
    C = c\, F, \qquad P = p\, F
\end{equation}

Quantflow uses forward space everywhere downstream of the input layer.
The `inverse` flag on [OptionPrice][quantflow.options.surface.OptionPrice]
only controls how the *input* `price` field is stored: for inverse
options (option premium paid in the underlying) it already is in forward
space; for non-inverse options (premium paid in the quote currency) it
is the absolute quote-currency price and must be divided by $F$ to enter
forward space. The
[price_in_forward_space][quantflow.options.surface.OptionPrice.price_in_forward_space]
property handles both cases uniformly.

## Hurst Exponent

The Hurst exponent is a measure of the long-term memory of time series. The Hurst exponent is a measure of the relative tendency of a time series either to regress strongly to the mean or to cluster in a direction.

Check this study on the [Hurst exponent with OHLC data](../applications/hurst).


## Kalman Filter

A recursive algorithm that estimates the latent state of a linear-Gaussian
[state-space model](#state-space-model) from noisy observations. Introduced in
[Kalman (1960)](bibliography.md#kalman). Given an
initial prior, the filter alternates between a *predict* step (advancing the
state through the linear dynamics) and an *update* step (incorporating a new
observation by Bayes' rule). Both steps preserve the Gaussian form, so the
filtering distribution is fully characterised by its mean and covariance.
When the observation noise covariance is a scaled identity and the observation
matrix is a column vector, the update can be accelerated via the
[Sherman-Morrison identity](#sherman-morrison-identity).
Implementation: [KalmanFilter][quantflow.ta.kalman.KalmanFilter].
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
and put options with the same strike $K$ and time to maturity.

\begin{equation}
    C - P = D_q \left(F - K\right)
\end{equation}

where $D_q$ is the [discount factor](#discount-factor) of the quoting asset (generally a currency)
at maturity and $F$ is the forward price
of the underlying asset at maturity.

Denoting forward-space prices
$c = C/(D_q\ F)$ and $p = P/(D_q\ F)$ (see [Black Pricing](api/options/black.md)), the relationship
reads:

\begin{equation}
    c - p = 1 - \frac{K}{F} = 1 - e^k
\end{equation}

where $k$ is the [log-strike](#log-strike).


## Sherman-Morrison Identity

A formula that computes the inverse of a matrix perturbed by a rank-1 outer
product in $O(n^2)$ time:

\begin{equation}
(A + u v^\top)^{-1} = A^{-1} - \frac{A^{-1} u v^\top A^{-1}}{1 + v^\top A^{-1} u}
\end{equation}

Used in the Kalman filter when the observation noise covariance is a scaled
identity $h^2 I$ and the observation matrix is a column vector $c$. The
innovation covariance $S = h^2 I + P\,c c^\top$ is then a rank-1 update to a
scaled identity, avoiding a full $O(n_y^3)$ solve.

## State-Space Model

A mathematical framework describing the joint evolution of an unobserved
(latent) state process $x_t$ and an observation process $y_t$ that depends on
the latent state:

\begin{equation}
\begin{aligned}
    x_t &\sim p(x_t \mid x_{t-1}) \\
    y_t &\sim p(y_t \mid x_t)
\end{aligned}
\end{equation}

See [Durbin & Koopman (2012)](bibliography.md#durbin_koopman) for a
comprehensive treatment.

Quantflow provides the abstract base
[StateSpaceModel][quantflow.ta.kalman.StateSpaceModel] and the concrete
[LinearGaussianModel][quantflow.ta.kalman.LinearGaussianModel] for the
linear-Gaussian case.
## Time To Maturity (TTM)

Time to maturity is the time remaining until an option or forward contract expires,
expressed in years. It is calculated using a day count convention applied to the
interval between the reference date and the expiry date. For a reference date $t_0$
and expiry date $T$:

$$\tau = \text{dcf}(t_0, T)$$

where $\text{dcf}$ is the day count fraction function (Act/Act by default in quantflow).
TTM is denoted $\tau$ throughout the pricing and calibration formulas.


## Unscented Kalman Filter (UKF)

An extension of the Kalman filter, introduced by
[Julier & Uhlmann (1997)](bibliography.md#julier_uhlmann), that uses the
[unscented transform](#unscented-transform) (a set of deterministically chosen
sigma points) to propagate distributions through non-linear transition
functions while maintaining a Gaussian approximation. The UKF generates $2n_x + 1$
sigma points around the current state mean, passes each through the model's
[transition_function][quantflow.ta.kalman.StateSpaceModel.transition_function],
and recomputes the predicted mean and covariance from the propagated points.
Implementation: [UnscentedKalmanFilter][quantflow.ta.kalman.UnscentedKalmanFilter].
