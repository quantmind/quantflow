# Discount Curves from Option Prices

Every implied volatility on a surface is inverted against a forward price.
The forward at each maturity is determined by the option market itself, and
getting it wrong corrupts every single volatility on the surface.

The central principle of this tutorial is simple: for a smooth implied
volatility surface, the forwards must be calculated from put-call parity.
The sections below show what goes wrong with any other forward, how parity
identifies the right one, and how the quantflow calibration implements the
principle: [calibrate_forwards][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_forwards]
estimates the forwards first, and
[calibrate_curves][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_curves]
derives the discount curves from them.
Everything is demonstrated on a recorded snapshot of the BTC option book from
[Deribit][quantflow.data.deribit.Deribit]. It is the reference for the
calibration design going forward. See
[Forwards and Discount Factors](../theory/forwards.md) for the underlying
theory.

## The data

The snapshot was recorded on 1 August 2026 and is rebuilt offline with
[loader_from_book][quantflow.data.deribit.Deribit.loader_from_book]. The
exact reference time appears in the chart titles. It contains 13 maturities,
a listed future at every maturity, and the perpetual as spot. Several hundred
option quotes carry both the call and the put side, so
[put-call parity](../glossary.md#put-call-parity) can be observed directly.

## The problem: a wrong forward

Implied volatilities are inverted from out of the money quotes: puts below the
forward, calls above.

An option price is the sum of intrinsic value and time value. Only the time
value carries volatility information. An out of the money option has no
intrinsic value, so its whole price is time value.

An in the money option still has meaningful time value near the money. But
the deeper it moves in the money, the more intrinsic value dominates its
price. The inversion then extracts a small time value from a large price, and
quote noise swings the implied volatility. By put-call parity the out of the
money option at the same strike carries the same information, with far better
conditioning. Market liquidity concentrates out of the money for the same
reason.

The forward therefore matters twice. It enters the Black formula, and before
that it decides which side of each strike is out of the money. The two sides
meet at the forward. If the inversion uses a forward that disagrees with the
option market, they disagree exactly there: with a forward that is too low,
the calls come out too high and the puts too low.

The chart below inverts the same mid quotes twice. The left panel uses the
spot price as forward and the smile breaks at the switch. The right panel
uses the listed future and the smile is continuous.

[![Smile with wrong and correct forward](../assets/examples/curve_calibration_smile.png)](../assets/examples/curve_calibration_smile.png){target="_blank"}

The jump is the forward error expressed in volatility units, so it grows with
the time to maturity. The spot is an extreme case, but the mechanism is
general. Any forward that deviates from the one embedded in the option quotes
leaves a jump at the switch, in proportion to the deviation. The question is
therefore how to find the forward the option market itself is using. That is
the fix section below.

**What the library does.**
[calibrate_forwards][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_forwards]
estimates the parity forward of every maturity and stores it on the cross
section. The surface then prices each cross section off its
[pricing_forward][quantflow.options.surface.VolCrossSection.pricing_forward]:
the parity forward when calibrated, the market forward of the cross section
otherwise. Discount curves are never used to derive pricing forwards.

**The check to remember.** A jump in the smile where puts switch to calls is
always the signature of a wrong forward. A correct calibration makes the two
sides meet, whatever curve model is used.

## Quote selection

The choice of which side to invert at each strike is controlled by
[OptionSelection][quantflow.options.surface.OptionSelection].

[`OTM`][quantflow.options.surface.OptionSelection.OTM] selects the out of
the money side, with the switch at the forward.
[`CALL`][quantflow.options.surface.OptionSelection.CALL],
[`PUT`][quantflow.options.surface.OptionSelection.PUT] and
[`ALL`][quantflow.options.surface.OptionSelection.ALL] select fixed sides
and are mostly diagnostic tools.
Comparing the `CALL` and `PUT` smiles at the same strikes is another way to
detect a wrong forward.

[`BEST`][quantflow.options.surface.OptionSelection.BEST] currently
selects the out of the money side and blends the call and
put implied volatilities near the money. The blending smooths a small
residual basis, but it cannot repair a wrong forward.

This tutorial assumes plain out of the money selection throughout.

**What remains open.** Placing the switch at the raw forward is itself a
simplification. The natural coordinate for the boundary is the
[vol adjusted moneyness](../glossary.md#moneyness-vol-adjusted), or its
[convexity adjusted](../glossary.md#moneyness-convexity-adjusted) variant.
Both measure the distance from the money in units of standard deviation.
They require the implied volatility, which is not known before the first
inversion. A solid `BEST` strategy therefore needs two passes: invert with
the plain out of the money selection first, then re-select the quotes using
the fitted volatilities and invert again. This strategy still needs to be
devised, implemented, and documented here.

## The fix: forwards from parity

The forward the option market is using is written in the option quotes
themselves. For every strike $K$ quoted on both sides,
[put-call parity](../glossary.md#put-call-parity) links the mid prices to the
two discount factors:

\begin{equation}
    C - P = D_a S - D_q K
\end{equation}

Within one maturity this is a straight line in the strike. The chart below
shows the regression at two maturities of the snapshot: the front maturity on
the left and a long maturity on the right, each titled with its date and time
to maturity. The dashed vertical lines mark where each regression line
crosses zero, which is the parity implied forward.

[![Put-call parity regression](../assets/examples/curve_calibration_parity.png)](../assets/examples/curve_calibration_parity.png){target="_blank"}

The long maturity is textbook: dozens of pairs, a strike range spanning three
times the spot, and quotes that sit exactly on the line. The front maturity
is a preview of the problems ahead: a handful of pairs, a strike range of a
few percent, and visible scatter around the line.

The **zero crossing** is the strike at which the parity line crosses zero,
visible in the chart above where the regression line meets the horizontal
axis. At that strike the call and the put have the same price. Setting
$C - P = 0$ in the equation above gives $K = S D_a / D_q$, which is the
[forward](../glossary.md#forwards) $F$.

This strike sits inside the quoted range, so it is found by interpolating
between quotes, never by extrapolating. The parity implied forward is
therefore extremely well identified. And it does not have to agree with the
listed future.

The term structure below compares the two, with the time to maturity on a
log scale to spread out the front.

[![Forward term structure](../assets/examples/curve_calibration_forwards.png)](../assets/examples/curve_calibration_forwards.png){target="_blank"}

Not all parity pairs deserve the same trust. The same regressions are shown
below in [moneyness](../glossary.md#moneyness) terms, computed against the
spot since the forward is not known yet. Moneyness measures the distance from
the money in square root of time units, so the two maturities become
comparable on this axis.

[![Parity in moneyness space](../assets/examples/curve_calibration_parity_moneyness.png)](../assets/examples/curve_calibration_parity_moneyness.png){target="_blank"}

A pair far from the money contains one option that is deep in the money. Its
quote is dominated by intrinsic value, tick rounding and bid ask spread, for
the same reasons discussed for the smile inversion. Those pairs drag the
regression away from the money, which is exactly where the zero crossing
lives. The shaded band keeps only pairs near the money.

**What the library does.**
[calibrate_forward][quantflow.options.parity.PutCallParities.calibrate_forward]
implements this as an iterative weighted regression. Each pair is weighted by
the inverse of its parity bid ask spread, so noisy pairs count less: this is
the data driven core of the estimator. On top of the weights, pairs are
selected inside a band of
[convexity adjusted moneyness](../glossary.md#moneyness-convexity-adjusted),
one standard deviation by default, widened automatically when too few pairs
survive. The band coordinates require the forward and the volatility, so the
algorithm iterates: fit the crossing, estimate the at the money volatility
from the nearest straddle, re-select, refit. Two or three iterations
suffice.

[calibrate_forwards][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_forwards]
runs the estimator at every maturity and stores the forwards on the cross
sections, where the surface picks them up for pricing.

The chart below shows the result: a naive regression over all pairs drifts
from the listed futures by up to half a percent, while the calibrated
forwards agree with them within a few basis points at every maturity,
including the noisy front.

[![Forward basis](../assets/examples/curve_calibration_forward_basis.png)](../assets/examples/curve_calibration_forward_basis.png){target="_blank"}

The choice of the band width is not critical, which is the point of the
weighted design. On this snapshot the maximum error against the listed
futures moves only between 4 and 8 basis points as the band varies from half
to three standard deviations, against 48 basis points for the unweighted all
pairs regression.

## The discount factor split

The parity line pins down the forward with high precision. The **split** of
the line into $D_a$ and $D_q$ is a different story. It requires the intercept
$D_a S$, which lives at $K = 0$, far outside any quoted strike. Extrapolating
there is ill-conditioned. Small quote noise moves both discount factors a
lot, in opposite directions.

**What the library does.**
[calibrate_curves][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_curves]
never fits the two discount factors freely. The forwards are calibrated
first and held fixed. With the forward known, put-call parity collapses to
a single free parameter, the quote discount factor:

\begin{equation}
    C - P = D_q \left(F - K\right)
\end{equation}

[quote_discount][quantflow.options.parity.PutCallParities.quote_discount]
estimates $D_q$ at each maturity by weighted least squares across the
parity pairs, with the same inverse spread weights used for the forward.
Fixing the forward removes the ill conditioning of the split: the slope is
the only parameter left and the crossing no longer moves.

For options traded on exchanges that settle without discounting, such as
Deribit, the quote discount factor is not an estimate but a market
convention: $D_q = 1$ at every maturity. In that case the quote curve is a
[NoDiscountCurve][quantflow.rates.no_discount.NoDiscountCurve] and is kept
as given: no fitting is required.

The asset discount factor is never estimated independently. It follows
from the forward formula:

\begin{equation}
    D_a = D_q \frac{F}{S}
\end{equation}

The asset curve is therefore always fitted to these discount factors, even
when the quote curve is a no discount curve: the entire forward to spot
basis is attributed to the asset leg, where it encodes the implied funding
rate of the asset.

The selected curve models are fitted to those discount factors: a
parametric model pools all maturities while an interpolated curve passes
through them. This guarantees that the curves are consistent with the parity
forwards, and since the surface prices off the parity forwards directly, the
smile cannot break whatever curve model is selected.

## Short maturities

The forward of a short maturity is well identified: the crossing is an
interpolation and the front forwards above agree with the futures within
basis points. The discounting of a short maturity is not. The quote discount
factor is the slope of the parity line, and a chain that expires in hours
quotes strikes only a few percent apart, so the slope estimate carries an
error of percents. In rate units the damage is then amplified:

\begin{equation}
    r(\tau) = -\frac{\ln D(\tau)}{\tau}
\end{equation}

A two percent discount factor error at three weeks is already a forty
percent zero rate error. This is not an estimation problem to fix: a chain
of short dated options simply contains no information about discounting.

**What the library does.**
[calibrate_curves][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_curves]
excludes maturities below `min_ttm` (two weeks by default) from the curve
fitting. Their forwards are still calibrated and used for pricing; their
discounting extrapolates from the first calibrated node, which is essentially
exact since the discount factor of a few days is one within basis points.

The chart below shows the calibrated zero rates. The interpolated nodes
scatter at the short end even after the exclusion, while the parametric
curves pool the maturities. Rates implied from options become meaningful
roughly beyond a month.

[![Parametric against interpolated rates](../assets/examples/curve_calibration_rates.png)](../assets/examples/curve_calibration_rates.png){target="_blank"}

## Summary

| Problem | Effect | Solution |
|---|---|---|
| Wrong forward | Smile jumps at the put to call switch | Forwards calibrated from parity, used directly for pricing |
| Parity pairs far from the money | Deep in the money leg drags the implied forward | Inverse spread weights and an adaptive moneyness band |
| Weak identification of the split | Freely fitted curves absorb quote noise | One parameter per maturity with the forward held fixed |
| Short maturity discounting | Front zero rates unidentifiable | Maturities below `min_ttm` excluded from curve fitting |
| Switch at the raw forward | Selection ignores the volatility scale | Open: two pass selection in vol adjusted moneyness |

The only open item is the two pass `BEST` selection of the quote selection
section. Everything else in this table is implemented by
[calibrate_forwards][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_forwards]
and
[calibrate_curves][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_curves].

## Code

```python
--8<-- "docs/examples/curve_calibration.py"
```

```
--8<-- "docs/examples/output/curve_calibration.out"
```
