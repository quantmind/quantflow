# Volatility Surface

This tutorial covers the full workflow for building an implied volatility surface:
fetching option quotes from Deribit, extracting implied forwards and discount factors
from option prices, and inspecting the surface inputs.

## Fetching Data from Deribit

The [Deribit][quantflow.data.deribit.Deribit] client exposes a high-level
[volatility_surface_loader][quantflow.data.deribit.Deribit.volatility_surface_loader]
method that fetches all option quotes for a given asset and assembles them into a
[VolSurfaceLoader][quantflow.options.surface.VolSurfaceLoader]:

```python
import asyncio
from quantflow.data.deribit import Deribit

async def load():
    async with Deribit() as cli:
        loader = await cli.volatility_surface_loader("btc")
    return loader

loader = asyncio.run(load())
```

Key parameters of `volatility_surface_loader`:

| Parameter | Default | Description |
|---|---|---|
| `asset` | required | Underlying asset, e.g. `"btc"`, `"eth"`, `"sol"` |
| `inverse` | `True` | Inverse options (settled in the underlying) |
| `use_perp` | `False` | Derive spot from the perpetual contract |
| `exclude_open_interest` | `0` | Drop strikes with open interest below this threshold |

## Building the Surface

The loader holds the raw market data. Call
[surface()][quantflow.options.surface.GenericVolSurfaceLoader.surface] to construct a
[VolSurface][quantflow.options.surface.VolSurface]:

```python
surface = loader.surface()
```

Then run [bs()][quantflow.options.surface.VolSurface.bs] to populate implied
volatilities via Black-Scholes inversion:

```python
surface.bs()
```

[bs()][quantflow.options.surface.VolSurface.bs] solves for the implied volatility that
matches each bid and ask price and marks each option as `converged` or not.

### Removing Outliers

Raw option quotes often contain illiquid or stale prices that produce unrealistic
implied volatilities.
[disable_outliers()][quantflow.options.surface.VolSurface.disable_outliers] removes
them in two passes per maturity.

```python
surface.disable_outliers()
```

## Inspecting Surface Inputs

The examples below use a saved snapshot of a real ETH surface. The workflow is identical
for a live surface fetched from Deribit.

```python
--8<-- "docs/examples/vol_surface_inputs.py"
```

[term_structure()][quantflow.options.surface.VolSurface.term_structure] shows forward
prices and the interest rate implied by the forward-spot basis for each maturity. The
option inputs table lists the bid/ask prices together with the corresponding implied
volatilities for each strike:

```
--8<-- "docs/examples/output/vol_surface_inputs.out"
```

## Serialising and Restoring

[inputs()][quantflow.options.surface.VolSurface.inputs] serialises the surface to a
[VolSurfaceInputs][quantflow.options.inputs.VolSurfaceInputs] object — a list of
[SpotInput][quantflow.options.inputs.SpotInput],
[ForwardInput][quantflow.options.inputs.ForwardInput], and
[OptionInput][quantflow.options.inputs.OptionInput] records — that can be stored or
transmitted as JSON and later reconstructed via
[surface_from_inputs][quantflow.options.surface.surface_from_inputs]:

```python
from quantflow.options.surface import surface_from_inputs

inputs = surface.inputs(converged=True)   # VolSurface -> VolSurfaceInputs
surface2 = surface_from_inputs(inputs)    # VolSurfaceInputs -> VolSurface
```

## Extracting Forwards and Discount Factors

Pricing an option requires two market inputs beyond the option price itself: the forward
price $F$ of the underlying at expiry, and the quote discount factor $D_q$ for that
maturity, where the quote currency is usually USD.

In liquid markets these quantities are directly observable. Futures and forward contracts
give $F$ outright, and interest rate swaps or government bond strips give $D_q$. In many
option markets, however, neither is quoted directly.

!!! note "Deribit Forward"

    Crypto options on Deribit are a clear example. There is no liquid term structure
    of interest rates, and while Deribit quotes futures for each expiry, they are
    often illiquid, with wide bid-ask spreads and stale or outright wrong prices.

    The forward for each expiry must therefore be inferred from the options
    themselves.

Even when forwards are available, the discount factor used to value options may differ
from the rate implied by the forward-spot basis. For equity options the carry includes
dividends and repo costs that are not captured by a simple interest rate curve. For
crypto inverse options the discount factor reflects funding in the underlying asset
rather than in dollars.

For these reasons, quantflow extracts $D_q$ and $D_a$ directly from the market prices
of options using [put-call parity](../glossary.md#put-call-parity).

### Put-call parity and the implied forward

For each maturity, the parity relationship is fitted in the normalized form:

\begin{equation}
\frac{C - P}{S} = D_a - D_q \frac{K}{S}
\end{equation}

where $S$ is the spot price and $D_a$ the asset discount factor. The same equation
holds for inverse options with the left hand side replaced by $c - p$, the price
difference in units of the underlying.

The price difference is linear in the strike, so a regression across strikes
identifies $D_a$ and $D_q$, and the line crosses zero exactly at the forward:

\begin{equation}
F = S \frac{D_a}{D_q}
\end{equation}

[put_call_parities][quantflow.options.surface.VolCrossSectionLoader.put_call_parities]
collects the most liquid pairs at each maturity, ranked by the bid-ask spread of the
parity price, and
[calibrate_forward][quantflow.options.parity.PutCallParities.calibrate_forward] fits the
regression and returns the implied forward.

### Discount curve calibration

The
[calibrate_curves][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_curves]
method builds the discount curves on top of the calibrated forwards. With the
forward of each maturity held fixed, put-call parity identifies the quote
discount factor $D_q$ as its only remaining parameter, and the asset discount
factor follows from the forward formula $D_a = D_q F / S$.

- **Quote curve**: pass a [YieldCurve][quantflow.rates.yield_curve.YieldCurve]
  type for `quote_curve` to fit it to the per maturity quote discount factors.
  Leave it as `None` to keep the current quote curve and treat it as known:
  this is the setup for exchanges that settle without discounting, such as
  Deribit, where the quote curve is a
  [NoDiscountCurve][quantflow.rates.no_discount.NoDiscountCurve] with
  $D_q = 1$ at every maturity.
- **Asset curve**: always fitted to the discount factors $D_a = D_q F / S$. It
  cannot be a no discount curve, since the parity forwards and the quote curve
  define it.

See the [curve calibration tutorial](curve_calibration.md) for how the
forwards and the discount factor split are estimated.
