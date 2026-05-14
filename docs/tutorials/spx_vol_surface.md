# SPX Volatility Surface

Build an implied volatility surface for the S&P 500 index from a Yahoo Finance
option chain, then calibrate a two-factor BNS model to it.

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

## BNS2 calibration

A single-factor diffusive Heston struggles on SPX because the short-dated
skew is too steep to absorb with a single mean-reversion timescale.
[BNS2][quantflow.sp.bns.BNS2] adds a second Gamma-OU variance factor and
injects jumps directly into the variance process, with the leverage parameter
mirroring those jumps into the log-price.

[BNS2Calibration][quantflow.options.calibration.bns.BNS2Calibration] fits
nine parameters with both factors sharing the same Gamma stationary marginal,
following the BNS superposition-of-OU construction. See the
[BNS tutorial](bns_calibration.md) for the full parameterisation and the
rationale behind tying $(\theta, \beta)$.

The initial parameters seed a fast factor ($\kappa = 20$) and a slow factor
($\kappa = 0.3$). Both leverages start negative to reflect the persistent
equity-style downside skew across the term structure. Residuals are scored in
implied-vol space ([ResidualKind.IV][quantflow.options.calibration.base.ResidualKind])
to weight the wings comparably to the ATM region.

### Calibrated parameters

--8<-- "docs/examples/output/spx_vol_surface.out"

[![SPX BNS2 calibrated smile](../assets/examples/spx_vol_surface_bns2.png)](../assets/examples/spx_vol_surface_bns2.png){target="_blank"}

The weight collapses almost entirely onto the fast factor, so $v_0$ for bns1
sits close to the ATM variance read off the 3D surface and bns2 contributes
only marginally to the variance level. The slow factor instead carries the
stronger negative leverage ($\rho \approx -0.84$ against the fast factor's
$\rho \approx -0.43$): its low $\kappa$ keeps jumps persistent in the
log-price, which is what shapes the long-dated downside skew. Both factors
share the same BDLP intensity and jump decay by construction.

The remaining short-maturity gap is structural to BNS, as discussed in the
[BNS tutorial](bns_calibration.md): jumps live in the variance process, so
the log-price wings are bounded by the variance jumps scaled by $|\rho_i|$.

## Code

```python
--8<-- "docs/examples/spx_vol_surface.py"
```
