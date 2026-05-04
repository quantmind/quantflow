# BNS Volatility Model

This tutorial calibrates the [BNS][quantflow.sp.bns.BNS] stochastic-volatility
model (Barndorff-Nielsen and Shephard) to an implied volatility surface, using
the same workflow as the Heston tutorial in
[Volatility Surface](volatility_surface.md).

BNS is structurally different from Heston. The variance process is a
non-Gaussian Ornstein-Uhlenbeck process driven by a pure-jump Lévy process
(Gamma-OU in this implementation), and the leverage effect is introduced by
correlating the same jumps into the log-price.

## Model Parameters

[BNSCalibration][quantflow.options.calibration.bns.BNSCalibration] fits five
parameters to the surface:

| Parameter | Description |
|---|---|
| `v0` | Initial variance ($v_0$) |
| `theta` | Long-run variance ($\theta = \lambda / \beta$) |
| `kappa` | Mean reversion speed of the variance process |
| `beta` | Exponential decay rate of the BDLP jump-size distribution |
| `rho` | Leverage parameter (correlation between jumps in variance and log-price) |

The BDLP intensity is set as $\lambda = \theta \beta$ so that the stationary
mean of the Gamma-OU variance process equals $\theta$. This gives the same
$(v_0, \theta)$ parameterisation as Heston.

Because the variance is built from positive jumps and exponential mean
reversion, it stays positive by construction. No Feller-style penalty is
needed.

## How BNS Fits the Surface

The mechanism that produces a smile in BNS is structurally different from
Heston. Heston relies on a diffusive volatility-of-variance $\sigma$ for the
wings and a spot-variance correlation $\rho$ for the skew, both accumulating
as $\sqrt{T}$.

BNS instead injects discrete jumps directly into the variance process: each
jump in $v_t$ is mirrored, scaled by $\rho$, into the log-price. The wing
thickness is governed by the jump-size distribution (controlled by $\beta$)
and the skew by $\rho$.

A consequence of this structural difference is that the calibrator often
settles at a small $\kappa$ together with a large $\theta$. The time scale of
mean reversion is $1/\kappa$, so when $\kappa$ is small the variance process
barely relaxes towards $\theta$ over the calibration horizon and stays close
to $v_0$ throughout.

In that regime $\theta$ is only weakly identified by the surface and the
optimizer can move it freely as long as the jump-driven smile dynamics are
preserved. The headline number to read in the output is $v_0$, which sets the
at-the-money level.

## Calibration

The fit reuses [VolModelCalibration][quantflow.options.calibration.base.VolModelCalibration]
two-stage optimiser from the Heston tutorial: L-BFGS-B for basin search,
followed by trust-region reflective on the residual vector with parameter
bounds.

--8<-- "docs/examples/output/vol_surface_bns_calibration.out"

## Calibrated Smile

[![BNS calibrated smile](../assets/examples/bns_calibrated_smile.png)](../assets/examples/bns_calibrated_smile.png){target="_blank"}

The fit is good for medium and long maturities and visibly off at the front
expiries. This is the same short-maturity gap seen for Heston and
Heston-jump-diffusion.

The cause here is structural: BNS adds jumps, but they live in the variance
process, not directly in the log-price. The jump-driven contribution to the
log-price is bounded by the size of the variance jumps multiplied by $|\rho|$,
which is small for short tenors.

A model with explicit jumps in the log-price (such as
[HestonJ][quantflow.sp.heston.HestonJ]) or a rough volatility model is better
suited to the steep short-term skew observed in crypto markets.

## Code

```python
--8<-- "docs/examples/vol_surface_bns_calibration.py"
```
