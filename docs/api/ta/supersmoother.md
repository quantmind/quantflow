# Super Smoother

SuperSmoother algorithm for time series smoothing.

Implementation based on John Ehlers' SuperSmoother filter,
which is a [two-pole Butterworth filter](https://en.wikipedia.org/wiki/Butterworth_filter) that provides excellent smoothing
while minimizing lag. The filter uses an adaptive approach with cross-validation
to determine the optimal smoothing parameters.

Reference:
    Ehlers, J. (2013). "Cycle Analytics for Traders"


::: quantflow.ta.SuperSmoother
