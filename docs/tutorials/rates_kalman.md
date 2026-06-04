# Yield Curve Calibration from Rates

This tutorial calibrates two short-rate models, the
[VasicekCurve][quantflow.rates.vasicek.VasicekCurve] and the
[CIRCurve][quantflow.rates.cir.CIRCurve], to a panel of historical US Treasury
yields by maximum likelihood. Both treat the short rate as a latent state and
the yields as noisy observations of it, but they need different filters: Vasicek
is linear-Gaussian and uses the exact
[KalmanFilter][quantflow.ta.kalman.KalmanFilter], while CIR has a state-dependent
variance and uses the
[UnscentedKalmanFilter][quantflow.ta.kalman.UnscentedKalmanFilter].

For the mechanics of the filters themselves, see the
[Kalman Filter](../theory/kalman.md) theory page.

## The idea

A short-rate model never observes its state directly: what we observe at each
date is a cross section of yields. Treating the short rate as a latent state and
the yields as noisy observations of it turns calibration into a state-space
estimation problem.

At each date every yield is affine in the short rate, so the observation is
linear in both models. The filter computes the Gaussian log-likelihood of the
observed panel, and the calibrator maximises it over
$(\kappa, \theta, \sigma, h)$, where $h$ is the observation noise standard
deviation.

## Fetching the data

[FederalReserve.yield_curves][quantflow.data.fed.FederalReserve.yield_curves]
returns the daily Treasury par-yield panel, indexed by date with one column per
tenor and rates as decimals. The `cached_df` helper stores the result as a
parquet file so repeated runs do not hit the network.

The calibration assumes a uniform time step, while the raw data is sampled on
business days. Resampling to weekly Wednesdays with the average yield over each
week gives an evenly spaced panel.

## Vasicek: the exact Kalman filter

The Vasicek short rate is an Ornstein-Uhlenbeck process. Over a uniform time step
its dynamics reduce to a Gaussian AR(1) with a constant innovation variance, and
each yield is affine in the short rate. The
[calibrate_historical_rates][quantflow.rates.vasicek.VasicekCurveCalibration.calibrate_historical_rates]
method documents these equations.

Because the model is fully linear-Gaussian, the exact Kalman filter gives the
exact log-likelihood. One full Kalman pass over the panel is performed per
optimiser iteration.

## CIR: why the unscented filter

The CIR short rate is a square-root diffusion,
$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$. Its conditional mean
is still linear in the previous state, but its conditional variance depends on
the state:

\begin{equation}
    \text{Var}[r_t \mid r_{t-1}] =
        r_{t-1}\,\frac{\sigma^2}{\kappa}\left(\phi - \phi^2\right)
        + \theta\,\frac{\sigma^2}{2\kappa}\left(1 - \phi\right)^2,
        \quad \phi = e^{-\kappa \Delta t}.
\end{equation}

The exact Kalman filter assumes a constant process-noise covariance, so it
cannot represent this heteroskedasticity. The unscented Kalman filter only needs
the conditional mean and covariance of the transition, which it propagates
through sigma points, so the state-dependent variance drops straight in. The
[CIRStateSpaceModel][quantflow.rates.cir.CIRStateSpaceModel] supplies those
moments and the affine observation, and
[calibrate_historical_rates][quantflow.rates.cir.CIRCurveCalibration.calibrate_historical_rates]
runs the unscented filter inside the same maximum-likelihood loop.

## Calibrating

For both models,
[calibrate_historical_rates_dataframe][quantflow.rates.calibration.YieldCurveCalibration.calibrate_historical_rates_dataframe]
parses the tenor columns into times to maturity, infers the time step from the
index, converts the par yields to continuously compounded rates (here
`frequency=2` for semiannual compounding), and runs the maximum-likelihood fit.

The fitted parameters and the final filtered short rate are returned on each
calibrated curve:

```
--8<-- "docs/examples/output/rates_kalman.out"
```

## Model versus observed through time

The fit is a time-series fit, so the right check is whether each model tracks the
history of every tenor. Both calibrators expose a `filtered_short_rate` path
([Vasicek][quantflow.rates.vasicek.VasicekCurveCalibration.filtered_short_rate],
[CIR][quantflow.rates.cir.CIRCurveCalibration.filtered_short_rate]), from which
each tenor's model-implied yield is reconstructed through the affine yield
relation and plotted against its observed history.

Both single-factor models track the short and intermediate tenors (1Y, 2Y)
closely and are smoother than the data at the long end (5Y, 10Y): one
mean-reverting factor cannot capture the independent variation of the long end.
The two fits stay close in this rate regime, the difference being the CIR
volatility that scales with the level of rates.

[![Observed vs Vasicek and CIR model yields](../assets/examples/rates_kalman.png)](../assets/examples/rates_kalman.png){target="_blank"}

## Code

```python
--8<-- "docs/examples/rates_kalman.py"
```
