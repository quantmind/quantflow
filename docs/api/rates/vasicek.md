# Vasicek Curve

The [VasicekCurve][quantflow.rates.vasicek.VasicekCurve] models the short rate and
yield curve under the Vasicek (Ornstein-Uhlenbeck) dynamics. Historical calibration
of model parameters is performed via maximum likelihood using the
[KalmanFilter][quantflow.ta.kalman.KalmanFilter] from [ta/kalman](../ta/kalman.md).

::: quantflow.rates.vasicek.VasicekCurve

::: quantflow.rates.vasicek.VasicekCurveCalibration
