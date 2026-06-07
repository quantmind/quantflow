# Heston Volatility Model

## Calibrating the Heston Model

[HestonCalibration][quantflow.options.calibration.heston.HestonCalibration] fits the
five Heston parameters ($v_0$, $\theta$, $\kappa$, $\sigma$, $\rho$) to the implied
volatility surface using a two-stage optimisation:

1. **L-BFGS-B** minimises the scalar cost function (sum of squared weighted price
   residuals) to reach a good basin of attraction.
2. **Trust-region reflective** (`least_squares` with `method="trf"`) refines the
   solution on the residual vector with tight tolerances and enforces parameter bounds.

Residuals are computed as `weight * (model_call_price - mid_call_price)` where
`mid_call_price` is the average of the bid and ask call prices.

The weight is $\min(e^{w \cdot m^2}, w_\text{max})$ controlled by
`moneyness_weight` (the coefficient $w$) and `max_cost_weight` (the cap
$w_\text{max}$), with $m = \log(K/F)/\sqrt{T}$ the standardised moneyness.
The quadratic exponent matches the gaussian shape of $1/\nu$ (inverse vega),
so a positive `moneyness_weight` puts wing residuals on the same footing as
ATM ones. The cap prevents a single deep-wing option from dominating the
loss.

A penalty for violating the Feller condition ($2\kappa\theta \geq \sigma^2$)
is added during stage 1 to keep the variance process well-behaved.

```python
--8<-- "docs/examples/vol_surface_heston_calibration.py"
```

### Output

--8<-- "docs/examples/output/vol_surface_heston_calibration.out"

### Calibration Options

The `moneyness_weight` parameter up-weights far-from-the-money options via
$e^{w \cdot m^2}$ where $m = \log(K/F)/\sqrt{T}$ is the standardised
moneyness. The result is capped at `max_cost_weight` (default 10) so a
single deep-wing option cannot dominate the loss.

### Plotting the Calibrated Smile

Use [plot_maturities()][quantflow.options.calibration.base.VolModelCalibration.plot_maturities]
to produce a Plotly figure overlaying market bid/ask implied vols against the model smile
for all maturities at once:

```python
fig = calibration.plot_maturities(max_moneyness_ttm=1.5, support=101)
fig.write_image("heston_calibrated_smile.png", width=1200)
```

The x axis is [moneyness](../glossary.md#moneyness).

![Heston calibrated smile](../assets/examples/heston_calibrated_smile.png)

### Model Limitations at Short Maturities

Inspecting the calibrated smiles across all maturities reveals a systematic pattern:
the Heston model fits long-dated options reasonably well but struggles with short-term
maturities, where the market smile is steeper than the model can reproduce.

This is a fundamental structural limitation, not a numerical issue. The Heston model
generates an implied volatility smile through two mechanisms: the correlation $\rho$
between spot and variance (which creates skew) and the volatility-of-variance $\sigma$
(which inflates the wings). Both effects accumulate diffusively over time. For a maturity
$T$, the smile roughly scales as $\sigma \sqrt{T}$, so as $T \to 0$ the distribution
collapses toward a Gaussian and the smile flattens.

More precisely, the Heston characteristic function at short maturities satisfies:

\begin{equation}
\log \phi(u, T) \approx i u \mu T - \tfrac{1}{2} u^2 v_0 T + O(T^2)
\end{equation}

which is the characteristic function of a Gaussian with variance $v_0 T$. The higher
cumulants that produce skew and excess kurtosis are all $O(T^2)$ or smaller, so they
vanish faster than the Gaussian term as $T \to 0$.

In practice this means the Heston model essentially reduces to Black-Scholes for
near-expiry options. The market, however, exhibits pronounced short-term skew driven by
jump risk and the market microstructure of short-dated hedging demand. A diffusion-only
model cannot reproduce this behaviour regardless of how its parameters are tuned.

The natural extension is to add a jump component to the dynamics, which contributes
a term of order $O(T)$ to the cumulants and restores the short-term smile. This is
the motivation for the Heston jump-diffusion model described in the next section.

## Calibrating the Heston Jump-Diffusion Model

[HestonJCalibration][quantflow.options.calibration.heston.HestonJCalibration] extends the
Heston calibration with a compound Poisson jump component via the
[HestonJ][quantflow.sp.heston.HestonJ] model. Jumps are drawn from a
[DoubleExponential][quantflow.dists.DoubleExponential] distribution,
which captures asymmetric jump behaviour common in equity and crypto markets.

```python
--8<-- "docs/examples/vol_surface_hestonj_calibration.py"
```

--8<-- "docs/examples/output/vol_surface_hestonj_calibration.out"

### Plotting the Calibrated Smile

```python
fig = calibration.plot_maturities(max_moneyness_ttm=1.5, support=101)
fig.write_image("hestonj_calibrated_smile.png", width=1200)
```

![HestonJ calibrated smile](../assets/examples/hestonj_calibrated_smile.png)

### Remaining Limitations at Short Maturities

Adding jumps improves the short-term smile significantly compared to plain Heston, but
the fit at the nearest maturities is still imperfect. Several structural reasons combine:

**Jump parameters are global.** The compound Poisson component has a single intensity
$\lambda$, jump variance, and asymmetry shared across all maturities. Increasing
$\lambda$ to steepen the short-term smile simultaneously distorts the long-term smile,
so the optimizer settles on a compromise.

**Long maturities dominate the cost function.** They have more liquid strikes and
therefore more data points. The optimizer minimizes total squared residuals across the
whole surface, so short maturities — with fewer strikes — are outvoted and their fit is
systematically sacrificed.

**The jump distribution is not rich enough.** The short-term smile in crypto is driven
by large, rare, asymmetric events. A [DoubleExponential][quantflow.dists.DoubleExponential]
with fixed parameters cannot simultaneously match the wing curvature at short and long
maturities.

The natural next step is a rough volatility model (for example rough Heston with Hurst
parameter $H < \tfrac{1}{2}$). Because the variance process has long memory and does not
behave diffusively at short time scales, rough models produce a steep short-term skew
without requiring jumps, and the skew decays as a power law $T^H$ rather than the
$T^{1/2}$ rate of classical stochastic volatility.

### Parameter Reference

The calibrated parameter vector for the jump-diffusion model is:

| Parameter | Description |
|---|---|
| `vol` | Initial volatility ($\sqrt{v_0}$) |
| `theta` | Long-run volatility ($\sqrt{\theta}$) |
| `kappa` | Mean reversion speed |
| `sigma` | Volatility of variance |
| `rho` | Spot-variance correlation |
| `jump intensity` | Jump arrival rate (jumps per year) |
| `jump variance` | Variance of a single jump |
| `jump asymmetry` | Asymmetry of the jump distribution ([DoubleExponential][quantflow.dists.DoubleExponential]) |
