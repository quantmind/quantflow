# Double Heston Calibration

This tutorial covers calibration of the
[DoubleHeston][quantflow.sp.heston.DoubleHeston] model to an implied volatility
surface. The Double Heston model drives a single log-price with two independent
CIR variance processes, each with its own mean-reversion speed, long-run variance,
vol-of-vol, and spot-variance correlation:

\begin{align}
    d x_t &= \sqrt{v^1_t}\,d w^1_t + \sqrt{v^2_t}\,d w^3_t \\
    d v^i_t &= \kappa_i (\theta_i - v^i_t) dt + \nu_i \sqrt{v^i_t}\,d w^{2i}_t
\end{align}

Because the two components are independent, the characteristic exponent is the sum
of the two individual Heston exponents, so no additional quadrature is needed.

## Motivation

A single Heston process produces a smile that flattens as $T \to 0$ (the smile
roughly scales as $\sigma\sqrt{T}$). Adding a second, faster-mean-reverting process
gives the model an extra degree of freedom to independently control the short-term
and long-term smile shapes without adding jumps.

The parameterisation enforces $\kappa_1 > \kappa_2$: the first process is the
short-maturity driver (fast mean-reversion) and the second is the long-maturity
driver (slow mean-reversion).

## Calibration

[DoubleHestonCalibration][quantflow.options.heston_calibration.DoubleHestonCalibration]
fits ten parameters jointly using the same two-stage optimisation as
[HestonCalibration][quantflow.options.heston_calibration.HestonCalibration]
(L-BFGS-B followed by trust-region reflective), but with a warm start that
initialises each process independently:

1. **Warm start**: fits a single Heston model to the long-dated options (ttm above
   the median) to initialise `heston2`, then fits a single Heston model to the
   short-dated options to initialise `heston1`.
2. **Stage 1**: L-BFGS-B on the ten-parameter joint cost function, with a Feller
   penalty applied independently to both variance processes.
3. **Stage 2**: trust-region reflective on the residual vector with tight tolerances
   and parameter bounds.

```python
--8<-- "docs/examples/vol_surface_double_heston_calibration.py"
```

### Output

--8<-- "docs/examples/output/vol_surface_double_heston_calibration.out"

### Calibrated Smile

![Double Heston calibrated smile](../assets/examples/double_heston_calibrated_smile.png)

## Adding Jumps: Double HestonJ

The pure diffusion Double Heston still struggles at the very short end because both
variance processes share the same $O(\sqrt{T})$ smile scaling. Adding a jump component
to the short-maturity process `heston1` restores an $O(T)$ contribution at small $T$
and gives the model the flexibility to match steep short-term skews.

[DoubleHestonJCalibration][quantflow.options.heston_calibration.DoubleHestonJCalibration]
extends the Double Heston calibration by attaching a compound Poisson jump process
(with a [DoubleExponential][quantflow.utils.distributions.DoubleExponential] jump
distribution) to `heston1` and adding the jump parameters to the optimisation vector.
The warm start fits a full
[HestonJCalibration][quantflow.options.heston_calibration.HestonJCalibration] to the
short-dated options so the jump parameters are also initialised before the joint fit.

```python
--8<-- "docs/examples/vol_surface_double_hestonj_calibration.py"
```

### Output

--8<-- "docs/examples/output/vol_surface_double_hestonj_calibration.out"

### Calibrated Smile

![Double HestonJ calibrated smile](../assets/examples/double_hestonj_calibrated_smile.png)
