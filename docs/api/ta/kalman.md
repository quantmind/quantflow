# Kalman Filter & Unscented Kalman Filter

This module provides generic Kalman and unscented Kalman filter
implementations built on an abstract
[state-space model](../../glossary.md#state-space-model).

It is influenced by the implementation of the State-Space model &
Kalman filter in the Python [particles](https://github.com/nchopin/particles) library.

A state-space model separates a time series into two layers:

* A **latent state** $x_t$ that carries the unobserved dynamics.
* An **observation** $y_t$ that is a (noisy) function of the current state.

See the [kalman filter](../../theory/kalman.md) theory page for more details
on the algorithms and their applications.

::: quantflow.ta.kalman.StateSpaceModel

::: quantflow.ta.kalman.LinearGaussianModel

::: quantflow.ta.kalman.KalmanFilter

::: quantflow.ta.kalman.UnscentedKalmanFilter
