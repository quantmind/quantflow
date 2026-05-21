# Interest Rates

The `quantflow.rates` module provides primitives for interest rate modelling: flat rates, yield curves, and curve fitting.

The central concept is the [discount factor](../../glossary.md#discount-factor) $D_\tau$, the present value of one unit of currency paid at time $\tau$. Every class in this module exposes a `discount_factor` method that computes $D_\tau$ from the configured rate or curve.

**[Rate](interest_rate.md)** represents a spot or forward interest rate with a chosen compounding frequency (continuous by default) and day count convention. It supports continuous and periodic compounding and can be bootstrapped directly from a spot/forward pair.

**[YieldCurve](yield_curve.md)** is the abstract base for term-structure models. It defines the interface via `discount_factor` and `instantaneous_forward_rate`, with the two quantities linked by

\begin{equation}
    f(\tau) = -\frac{\partial \ln D_\tau}{\partial \tau}
\end{equation}

**[NelsonSiegel](nelson_siegel.md)** is a concrete `YieldCurve` implementation that fits a smooth parametric curve to observed zero-coupon rates using the Nelson-Siegel functional form.

**[Options Discounting](options.md)** provides `YieldCurveCalibration`, the base class for fitting a yield curve to discount factors, and `OptionsDiscountingCalibration`, which bootstraps asset and quote curves from put-call parity observations.
