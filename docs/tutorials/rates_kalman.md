# Vasicek Calibration from Rates

This tutorial calibrates the [VasicekCurve][quantflow.rates.vasicek.VasicekCurve]
short-rate model to a panel of historical US Treasury yields by maximum
likelihood, using a [KalmanFilter][quantflow.ta.kalman.KalmanFilter] to evaluate
the likelihood. It shows the full workflow: pulling the data from the Federal
Reserve, reshaping it into a uniform panel, fitting the model, and comparing the
fitted curve with the observed yields.

For the mechanics of the filter itself, see the
[Kalman Filter](../theory/kalman.md) theory page.

## The idea

The Vasicek short rate is an Ornstein-Uhlenbeck process. We never observe it
directly: what we observe is a cross section of yields at each date. Treating
the short rate as a latent state and the yields as noisy linear observations of
it turns calibration into a state-space estimation problem.

Over a uniform time step the dynamics reduce to a Gaussian AR(1) and each yield
is affine in the short rate. The
[calibrate_historical_rates][quantflow.rates.vasicek.VasicekCurveCalibration.calibrate_historical_rates]
method documents these equations. The Kalman filter computes the exact Gaussian
log-likelihood of the observed panel, and the calibrator maximises it over
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

## Calibrating

[calibrate_historical_rates_dataframe][quantflow.rates.calibration.YieldCurveCalibration.calibrate_historical_rates_dataframe]
parses the tenor columns into times to maturity, infers the time step from the
index, converts the par yields to continuously compounded rates (here
`frequency=2` for semiannual compounding), and runs the maximum-likelihood fit.
One full Kalman pass over the panel is performed per optimiser iteration.

The fitted parameters and the final filtered short rate are returned on the
calibrated curve:

```
--8<-- "docs/examples/output/rates_kalman.out"
```

## Model versus observed through time

The fit is a time-series fit, so the right check is whether the model tracks the
history of each tenor. The calibrator exposes the
[filtered_short_rate][quantflow.rates.vasicek.VasicekCurveCalibration.filtered_short_rate]
path, from which each tenor's model-implied yield is reconstructed through the
affine yield relation and plotted against its observed history.

The single factor tracks the short and intermediate tenors (1Y, 2Y) closely, but
the model yields are smoother than the data at the long end (5Y, 10Y): one
mean-reverting factor cannot capture the independent variation of the long end,
the expected limitation of the one-factor Vasicek model.

[![Observed vs Vasicek model yields](../assets/examples/rates_kalman.png)](../assets/examples/rates_kalman.png){target="_blank"}

## Code

```python
--8<-- "docs/examples/rates_kalman.py"
```
