# Forwards and Discount Factors

## Discount Factors

A [discount factor](../glossary.md#discount-factor) $D(\tau)$ is the present value of
one unit of currency delivered at time $\tau$.

In quantflow, discount factors are provided by a
[YieldCurve][quantflow.rates.yield_curve.YieldCurve]. Different implementations capture
different term structures: a flat zero-rate curve, a fitted
[Nelson-Siegel][quantflow.rates.nelson_siegel.NelsonSiegel] curve, or any custom term
structure.

## Forward Price

The forward price of an asset at maturity $\tau$ is defined under the assumption that
two discount factors are available: one for the asset $D_a(\tau)$ and one for the
quote $D_q(\tau)$.

\begin{equation}
F(\tau) = S \cdot \frac{D_a(\tau)}{D_q(\tau)}
\end{equation}

where $S$ is the current spot price.

For an asset that pays no dividends and has no carry costs, the discount factor
for the asset is constant and equal to one. In this case
the forward price is simply given by the discount factor for the quote:

\begin{equation}
F(\tau) = \frac{S}{D_q(\tau)}
\end{equation}

When the quote is cash, the forward price is the spot price compounded at the risk-free rate,
which means that forward prices are typically higher than the spot price.

## Put-Call Parity

The Put call parity is a fundamental relationship between the prices of European call and put options with the same strike and maturity.
It states that the difference between the call price $C$ and the put price $P$ is equal to the discounted difference between the forward price $F$ and the strike price $K$:

\begin{equation}
C - P = S \cdot D_a - D_q \cdot K
\end{equation}

Dividing by the Spot price

\begin{equation}
\frac{C - P}{S} = D_a - \frac{D_q}{S} \cdot K
\end{equation}

This is linear in $K$, with slope $-D_q / S$ and intercept $D_a$.

Fitting a linear regression across multiple strikes at the same maturity therefore
identifies both discount factors simultaneously: $D_q$ from the slope and $D_a$ from
the intercept.

### Inverse Options

For inverse options, prices are quoted in units of the underlying asset, which acts as
the quote. The relevant discount factor is therefore $D_a$. Put-call parity becomes:

\begin{equation}
c - p = D_a \left(1 - \frac{K}{F}\right) = D_a - \frac{D_q}{S} \cdot K
\end{equation}

This is again linear in $K$, with intercept $D_a$ and slope $-D_q / S$, exactly the same as for regular options.
The same regression therefore identifies both discount factors as before.
