# SPX Volatility Surface

Build an implied volatility surface for the S&P 500 index from a Yahoo Finance
option chain.

The [Yahoo][quantflow.data.yahoo.Yahoo] client fetches the full chain for a
ticker. To keep this tutorial offline and reproducible we load a snapshot from
a gzipped JSON fixture, but the code is identical to what you would run
against the live endpoint.

## Loading the chain

[Yahoo.loader_from_chain][quantflow.data.yahoo.Yahoo.loader_from_chain] turns
the raw chain dictionary into a
[VolSurfaceLoader][quantflow.options.surface.VolSurfaceLoader]. SPX options
are non-inverse (quoted in USD) and Yahoo does not provide forwards, so each
maturity's forward is recovered from put-call parity inside the loader.

Once the loader has the data, [surface()][quantflow.options.surface.GenericVolSurfaceLoader.surface]
builds the [VolSurface][quantflow.options.surface.VolSurface],
[bs()][quantflow.options.surface.VolSurface.bs] inverts each bid and ask
through Black-Scholes, and
[disable_outliers()][quantflow.options.surface.VolSurface.disable_outliers]
drops strikes with unrealistic implied vols.

## 3D surface

[plot3d()][quantflow.options.surface.VolSurface.plot3d] renders the
converged implied vols against moneyness and time to maturity.

[![SPX implied volatility surface](../assets/examples/spx_vol_surface.png)](../assets/examples/spx_vol_surface.png){target="_blank"}

## Code

```python
--8<-- "docs/examples/spx_vol_surface.py"
```
