# Option Pricing

We use characteristic function inversion to price European call options on an underlying $S_t = S_0 e^{s_t}$, where $S_0$ is the spot price at time 0. We assume zero interest rates, so the forward equals the spot. The log-return $s_t = x_t - c_t$ is constructed from a driving process $x_t$ and a deterministic [convexity correction](convexity_correction.md) $c_t$ that enforces the martingale condition ${\mathbb E}[e^{s_t}] = 1$.

## Call Option

The price $C$ of a call option with strike $K$ is defined as

\begin{equation}
\begin{aligned}
C &= S_0 c_k \\
k &= \ln\frac{K}{S_0} \\
c_k &= {\mathbb E}\left[\left(e^{s_t} - e^k\right)^+\right]
    = \int_{-\infty}^\infty \left(e^s - e^k\right)^+ f_{s_t}(s)\, ds
\end{aligned}
\label{call-price}
\end{equation}

$k$ is the [log-strike](../glossary.md#log-strike) and $f_{s_t}$ is the probability density function of $s_t$. The call price is the discounted expected payoff under the risk-neutral measure, which simplifies to the undiscounted expected payoff when interest rates are zero.

All three methods share this starting point. They all express $c_k$ via the characteristic function $\Phi_{s_t}$, but differ in how the integration contour is chosen, how the payoff is handled, and the discretisation strategy.

## Carr & Madan

We follow [Carr & Madan](../bibliography.md#carr_madan) and write the Fourier transform of the call option as

\begin{equation}
\Psi_u = \int_{-\infty}^\infty e^{i u k} c_k dk
\end{equation}

Note that $c_k$ tends to $e^{s_t}$ as $k \to -\infty$, therefore the call price function is not square-integrable. In order to obtain integrability, Carr-Madan introduces a damping factor $e^{\alpha k}$ and works with the modified call $\tilde{c}_k = e^{\alpha k} c_k$. The free parameter $\alpha > 0$ is chosen so that $\tilde{c}_k$ is square-integrable. Taking the Fourier transform of $\tilde{c}_k$ and evaluating at $u = v - i\alpha$ (real $v$) gives

\begin{equation}
\Psi_u = \frac{\Phi_{s_t}\left(u-i\right)}{iu \left(1 + iu\right)}
\end{equation}

Inverting, the call price is recovered as

\begin{equation}
\begin{aligned}
c_k &= \int_0^{\infty} e^{-iuk} \Psi\left(u\right) du \\
    &= \frac{e^{-\alpha k}}{\pi} \int_0^{\infty} e^{-ivk} \Psi\left(v-i\alpha\right) dv \\
\end{aligned}
\end{equation}

The same FFT/FRFT machinery used for PDF inversion applies here.

### Choice of $\alpha$

For $\Psi_u$ to be well-defined, the characteristic function $\Phi_{s_t}(u - i)$ must be finite at $u = v - i\alpha$, i.e. $\Phi_{s_t}(v - i(\alpha + 1))$ must be finite. Denoting the upper edge of the strip of analyticity by $\beta_+$, this requires

\begin{equation}
\alpha + 1 < \beta_+
\end{equation}

Positive values of $\alpha$ assist the integrability of the modified call value over the
negative moneyness axis, but aggravate the same condition for the positive moneyness axis. For the modified call value to be integrable in the positive moneyness
direction, and hence for it to be square-integrable as well, a sufficient condition
is provided by $\Psi_{-i\alpha}$ being finite, which means the characteristic function $\Phi_{t,{-(\alpha+1)i}}$ is finite.

The constraint $\alpha + 1 < \beta_+$ can be restrictive: if the model has thin tails then $\beta_+$ is close to 1 and no positive $\alpha$ satisfies it cleanly. A poor choice of $\alpha$ leads to numerical instabilities, especially at short maturities.

## Lewis Formula

[Lewis](../bibliography.md#lewis) starts from the same call price expression but avoids the damping parameter by applying [Parseval's theorem](../glossary.md#parsevals-theorem) directly to $\eqref{call-price}$ and
shifting the integration in the complex plane to ensure the call options transform is well-defined and integrable.
The shift uses the [Residue theorem](https://en.wikipedia.org/wiki/Residue_theorem) to account for the poles of the payoff transform.

### Fourier Transform of the Payoff

The Fourier transform of the call payoff $g(s) = (e^s - e^k)^+$ is

\begin{equation}
\hat{g}(u) = \int_{-\infty}^\infty e^{ius} (e^s - e^k)^+\, ds = \int_k^\infty e^{ius}(e^s - e^k)\, ds
\end{equation}

For the first term to be integrable, we need $\mathrm{Im}(u) > 1$, while
for the second term we need $\mathrm{Im}(u) > 0$. Therefore the Fourier
transform of the payoff is well-defined in the strip $\mathrm{Im}(u) > 1$.

Splitting and integrating term by term gives

\begin{equation}
\hat{g}(u) = \frac{e^{(1+iu)k}}{iu(1+iu)} \qquad \mathrm{Im}(u) > 1
\label{payoff-ft}
\end{equation}

The denominator is zero at $u = 0$ and $u = i$.

This result is typical, option payoffs have a Fourier transform as long as we admit a complex valued transform variable and integrate along a horizontal line in the complex plane.

### Derivation

The call option can now be expressed via Parseval's theorem on a horizontal line in the complex plane.
We therefore evaluate the pricing integral along a horizontal contour

\begin{equation}
    u = v + i\eta \qquad v,\eta \in \mathbb{R}
\end{equation}

Since $\Phi$ is real-valued, we can replace $\overline{\Phi(u)}$ by $\Phi(-u)$:

\begin{equation}
\begin{aligned}
c_k &= \int_{-\infty}^\infty g(s) f(s) ds \\
&= \frac{1}{2\pi} \int_{-\infty}^\infty \hat{g}(u) \Phi_{s_t}(-u) du \\
&= \frac{e^k}{2\pi} \int_{-\infty}^\infty \frac{e^{iuk}}{iu(1+iu)} \Phi_{s_t}(-u) du
\end{aligned}
\end{equation}

The integral can be evaluated for any $\eta > 1$ as discussed in the previous section.
However, the integrand can now be computed for smaller $\eta$, provided the two `poles` at $u = 0$ and $u = i$ are avoided.

If we move the contour in $\eta \in (0,1)$, the integrand is well-defined. By the residue theorem, the value of
the integral is changed by the contribution of minus $2\pi i$ times the residues at $u = i$.

\begin{equation}
c_k = \frac{e^k}{2\pi} \int_{-\infty}^\infty \frac{e^{iuk}}{iu(1+iu)} \Phi_{s_t}(-u) du -2\pi i\, \text{Res}(i) \qquad \eta \in (0,1)
\end{equation}

The residue at $u = i$ is $\frac{i}{2\pi}$, so the call price can be expressed as

\begin{equation}
c_k = 1 + \frac{e^k}{2\pi} \int_{-\infty}^\infty \frac{e^{iuk}}{iu(1+iu)} \Phi_{s_t}(-u) du \qquad \eta \in (0,1)
\end{equation}

By choosing $\eta=\frac{1}{2}$, symmetrically located between the two poles, we can avoid numerical instabilities and obtain a stable formula for the call price:

\begin{equation}
c_k = 1 + \frac{e^k}{2\pi} \int_{-\infty}^\infty \frac{e^{iuk}}{iu(1+iu)} \Phi_{s_t}(-u) du \qquad u = v + \frac{i}{2}
\end{equation}

An substituting gives the final formula:

\begin{equation}
c_k = 1 + \frac{e^k}{2\pi} \int_{0}^\infty \frac{e^{i(v + i/2)k}}{i(v + i/2)(1+i(v + i/2))} \Phi_{s_t}(-v - \frac{i}{2}) dv
\end{equation}


### Comparison with Carr & Madan

The plots below use a Heston model to compare the two formulas and highlight where the
choice of $\alpha$ matters.

At a normal maturity (TTM=0.5) both methods agree closely with auto-selected $\alpha$:

![Carr-Madan vs Lewis prices](../assets/examples/lewis_vs_madan_prices.png)

At short maturities the auto-selected $\alpha$ in Carr & Madan can produce significant
errors, while Lewis remains stable:

![Short maturity difference](../assets/examples/lewis_vs_madan_short_ttm.png)

The sensitivity to the choice of $\alpha$ is most visible at TTM=0.02. A poor choice
(e.g. $\alpha=0.25$) yields completely wrong prices deep OTM, while Lewis (dotted) is
the stable reference:

![Alpha sensitivity](../assets/examples/lewis_vs_madan_alpha.png)

The example code that generates these plots:

```python
--8<-- "docs/examples/carr_madan_vs_lewis.py"
```

## COS Method

The [COS method](../bibliography.md#cos) (Fang and Oosterlee, 2008) takes a different
approach: rather than expressing the call price as a contour integral of the
characteristic function, it approximates the density $f_{s_t}$ on a truncated interval
$[a, b]$ using a cosine series and evaluates the payoff integral analytically against
each basis function.

### Density approximation

On the truncated interval $[a, b]$, the density is expanded as

\begin{equation}
    f_{s_t}(x) \approx \frac{2}{b-a}
    \sum_{j=0}^{N-1}{}^{\prime}
    \mathrm{Re}\!\left[
        \hat{\Phi}\!\left(\frac{j\pi}{b-a}\right) e^{-ij\pi a/(b-a)}
    \right]
    \cos\!\left(\frac{j\pi(x-a)}{b-a}\right)
\end{equation}

where $\hat{\Phi}$ is the martingale-corrected characteristic function and the prime
denotes a half weight on the $j=0$ term. The truncation interval is chosen as
$[a, b] = [-L\sigma, L\sigma]$ where $\sigma$ is the standard deviation of $s_t$ and
$L$ is a parameter (default 12).

### Payoff coefficients

Substituting the density approximation into the call price integral and changing
variables to $y = x - k$ turns the payoff into the strike-independent form
$(e^y - 1)^+$. The payoff integral against each cosine basis function evaluates
analytically as

\begin{equation}
    V_j = \frac{2}{b-a}\left[\chi_j(0,b) - \psi_j(0,b)\right]
\end{equation}

where $\nu_j = j\pi/(b-a)$ and

\begin{equation}
    \chi_j(0,b) = \int_0^b e^y \cos\!\left(\nu_j(y-a)\right) dy
    = \frac{(-1)^j e^b - \cos(\nu_j a) + \nu_j\sin(\nu_j a)}{1 + \nu_j^2}
\end{equation}

\begin{equation}
    \psi_j(0,b) = \int_0^b \cos\!\left(\nu_j(y-a)\right) dy
    = \begin{cases} b & j = 0 \\ \dfrac{\sin(\nu_j a)}{\nu_j} & j > 0 \end{cases}
\end{equation}

### Call price

Combining the density approximation with the payoff coefficients and accounting for the
change of variables gives the call price in forward space as

\begin{equation}
    c_k \approx e^k \sum_{j=0}^{N-1}{}^{\prime}
        \mathrm{Re}\!\left[
            \hat{\Phi}\!\left(\frac{j\pi}{b-a}\right)
            e^{-ij\pi(k+a)/(b-a)}
        \right] V_j
\end{equation}

The $e^k$ prefactor converts from the strike-normalised density of $y = x - k$ back to
forward-space pricing. Unlike Carr-Madan, no damping parameter is needed. Unlike Lewis,
the sum is over the payoff domain rather than the frequency domain, which can give
faster convergence for smooth densities at the cost of a fixed truncation error
controlled by $L$.
