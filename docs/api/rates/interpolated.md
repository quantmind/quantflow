# Interpolated Curves

Interpolated curves build the term structure directly from observed continuously compounded zero rates at a set of anchor dates, interpolating between them rather than fitting a parametric form.

The [InterpolatedYieldCurve][quantflow.rates.interpolated.InterpolatedYieldCurve] base class holds the interpolation nodes and derives the discount factor and instantaneous forward rate from the interpolated zero rate. Subclasses choose how the zero rate is interpolated between nodes.

[InterpolatedLinearCurve][quantflow.rates.interpolated.InterpolatedLinearCurve] interpolates the zero rate piecewise linearly, giving a forward rate that is linear on each segment.

[InterpolatedMonotonicCubicCurve][quantflow.rates.interpolated.InterpolatedMonotonicCubicCurve] uses a shape-preserving PCHIP cubic spline, giving a smooth zero rate and forward rate that introduces no spurious local extrema between nodes.

::: quantflow.rates.interpolated.InterpolatedYieldCurve

::: quantflow.rates.interpolated.InterpolatedLinearCurve

::: quantflow.rates.interpolated.InterpolatedMonotonicCubicCurve

::: quantflow.rates.interpolated.InterpolatedYieldCurveCalibration
