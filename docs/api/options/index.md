# Options

The `options` module provides classes and functions for pricing and calibrating options.

## Volatility Surface

The central class is [VolSurface][quantflow.options.surface.VolSurface], which represents
the implied volatility surface for an asset across all strikes and maturities. It holds:

- a [SpotPrice][quantflow.options.surface.SpotPrice] for the underlying asset
- a sorted tuple of [VolCrossSection][quantflow.options.surface.VolCrossSection] objects, one per maturity

Each [VolCrossSection][quantflow.options.surface.VolCrossSection] contains the forward price
at that maturity and a tuple of [Strike][quantflow.options.surface.Strike] objects.
Each [Strike][quantflow.options.surface.Strike] holds a call and/or put as an
[OptionPrices][quantflow.options.surface.OptionPrices], which in turn pairs a bid and ask
[OptionPrice][quantflow.options.surface.OptionPrice].

A surface is typically constructed via [VolSurfaceLoader][quantflow.options.surface.VolSurfaceLoader],
which accepts price inputs incrementally and builds the surface through its `surface()` method.
The lower-level [GenericVolSurfaceLoader][quantflow.options.surface.GenericVolSurfaceLoader]
provides the same functionality with a user-defined security type.

## Price Classes

| Class | Description |
|---|---|
| [Price][quantflow.options.surface.Price] | Base bid/ask price for any security |
| [SpotPrice][quantflow.options.surface.SpotPrice] | Spot bid/ask price of an underlying asset |
| [FwdPrice][quantflow.options.surface.FwdPrice] | Forward bid/ask price at a specific maturity |
| [OptionPrice][quantflow.options.surface.OptionPrice] | Single-sided option price with implied volatility and convergence flag |
| [OptionPrices][quantflow.options.surface.OptionPrices] | Paired bid and ask [OptionPrice][quantflow.options.surface.OptionPrice] for a given strike and option type |

## Input Classes

The input classes are plain data containers used to serialize and deserialize volatility surface data,
for example when storing or transmitting a snapshot of the surface.

| Class | Description |
|---|---|
| [VolSurfaceInputs][quantflow.options.inputs.VolSurfaceInputs] | Top-level container: asset name, reference date, and a list of inputs |
| [VolSurfaceInput][quantflow.options.inputs.VolSurfaceInput] | Base input with bid, ask, open interest and volume |
| [SpotInput][quantflow.options.inputs.SpotInput] | Input for a spot price |
| [ForwardInput][quantflow.options.inputs.ForwardInput] | Input for a forward price with maturity |
| [OptionInput][quantflow.options.inputs.OptionInput] | Input for an option with strike, maturity, type, and optional implied vols |

A [VolSurface][quantflow.options.surface.VolSurface] can be round-tripped via:

```python
inputs = surface.inputs()          # VolSurface -> VolSurfaceInputs
surface = surface_from_inputs(inputs)  # VolSurfaceInputs -> VolSurface
```
