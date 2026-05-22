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
price $F$ of the underlying at expiry, and the discount factor $D$ for that maturity.

In liquid markets these quantities are directly observable. Futures and forward contracts
give $F$ outright, and interest rate swaps or government bond strips give $D$. In many
option markets, however, neither is quoted directly. Crypto options on Deribit are a
clear example: there is no liquid term structure of interest rates and the forward for
each expiry must be inferred from the options themselves.

Even when forwards are available, the discount factor used to value options may differ
from the rate implied by the forward-spot basis. For equity options the carry includes
dividends and repo costs that are not captured by a simple interest rate curve. For
crypto inverse options the discount factor reflects funding in the underlying asset
rather than in dollars.

For these reasons, quantflow can extract $D_q$ and $D_a$ directly from the market prices
of options using put-call parity. The
[calibrate_curves][quantflow.options.surface.GenericVolSurfaceLoader.calibrate_curves]
method supports three modes:

- **Both curves**: pass a [YieldCurve][quantflow.rates.yield_curve.YieldCurve] type for
  both `quote_curve` and `asset_curve`. A single OLS regression per maturity identifies
  $D_q$ and $D_a$ simultaneously from the slope and intercept.
- **Asset curve only**: pass a type for `asset_curve` and leave `quote_curve` as `None`.
  The existing `quote_curve` on the loader is treated as known and $D_a$ is computed
  analytically from each put-call pair using the known $D_q$.
- **Quote curve only**: pass a type for `quote_curve` and leave `asset_curve` as `None`.
  The same simultaneous OLS is run but only the quote discount factors are used to fit
  the curve.
