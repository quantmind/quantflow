from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field, PrivateAttr
from scipy.interpolate import PchipInterpolator
from scipy.optimize import Bounds
from typing_extensions import Annotated, Doc

from quantflow.utils.dates import as_utc
from quantflow.utils.numbers import DecimalNumber
from quantflow.utils.types import FloatArray, FloatArrayLike, maybe_float

from .calibration import YieldCurveCalibration
from .yield_curve import YieldCurve

_YEAR = 365.0 * 86400.0


class InterpolatedYieldCurve(YieldCurve, arbitrary_types_allowed=True):
    r"""Base class for yield curves built by interpolating the zero rate.

    The curve is defined by continuously compounded zero rates $r_i$ at a set of
    anchor dates with times to maturity $\tau_i$ measured from
    [ref_date][.ref_date] on an ACT/365 basis. The zero rate is interpolated
    between the nodes and the discount factor follows directly:

    \begin{equation}
        D(\tau) = e^{-r(\tau)\,\tau}
    \end{equation}

    The instantaneous forward rate is then $f(\tau) = r(\tau) + \tau\,r'(\tau)$.
    Outside the node range the zero rate is held flat (the first node value below
    $\tau_1$ and the last node value beyond $\tau_N$).

    Subclasses choose how the zero rate is interpolated between nodes:
    [InterpolatedLinearCurve][..InterpolatedLinearCurve] (piecewise linear) or
    [InterpolatedMonotonicCubicCurve][..InterpolatedMonotonicCubicCurve]
    (shape-preserving PCHIP spline).

    The anchor lists default to empty, leaving the curve uncalibrated until its
    [calibrator][.calibrator] fills in the nodes.
    """

    anchor_dates: list[datetime] = Field(
        default_factory=list,
        description="Maturity dates of the interpolation nodes, strictly after the "
        "reference date and in increasing order",
    )
    anchor_rates: list[DecimalNumber] = Field(
        default_factory=list,
        description="Continuously compounded zero rates at each anchor date "
        "(0.05 means 5%), same length as anchor_dates",
    )

    _ttm: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))
    _rates: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))

    def model_post_init(self, context: object) -> None:
        """Cache the times to maturity and zero rates at the nodes."""
        if len(self.anchor_dates) != len(self.anchor_rates):
            raise ValueError("anchor_dates and anchor_rates must have equal length")
        if not self.anchor_dates:
            # uncalibrated curve: nodes are filled in by the calibrator
            self._ttm = np.empty(0)
            self._rates = np.empty(0)
            return
        ttm = self._year_fractions()
        if np.any(ttm <= 0):
            raise ValueError("anchor_dates must be strictly after the reference date")
        if np.any(np.diff(ttm) <= 0):
            raise ValueError("anchor_dates must be strictly increasing")
        self._ttm = ttm
        self._rates = np.array([float(r) for r in self.anchor_rates], dtype=float)

    def _year_fractions(self) -> FloatArray:
        """Times to maturity in years from ref_date, ACT/365."""
        ref = as_utc(self.ref_date)
        return np.array(
            [(as_utc(m) - ref).total_seconds() / _YEAR for m in self.anchor_dates],
            dtype=float,
        )

    def calibrator(self) -> InterpolatedYieldCurveCalibration:
        """Return an [InterpolatedYieldCurveCalibration][
        ...InterpolatedYieldCurveCalibration] wrapping this curve."""
        return InterpolatedYieldCurveCalibration(yield_curve=self)

    @abstractmethod
    def _zero_rate(
        self, tau: Annotated[FloatArray, Doc("Times to maturity, clamped to >= 0.")]
    ) -> FloatArray:
        """Interpolated zero rate, held flat outside the node range."""

    @abstractmethod
    def _zero_rate_derivative(
        self, tau: Annotated[FloatArray, Doc("Times to maturity, clamped to >= 0.")]
    ) -> FloatArray:
        """Derivative of the zero rate, zero outside the node range."""

    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        tau = np.maximum(np.asarray(ttm, dtype=float), 0.0)
        r = self._zero_rate(tau)
        dr = self._zero_rate_derivative(tau)
        return maybe_float(r + tau * dr)

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        tau = np.maximum(np.asarray(ttm, dtype=float), 0.0)
        r = self._zero_rate(tau)
        return maybe_float(np.exp(-r * tau))


class InterpolatedLinearCurve(InterpolatedYieldCurve):
    r"""Yield curve interpolating the zero rate piecewise linearly.

    The zero rate $r(\tau)$ is linear between adjacent nodes, so the
    instantaneous forward rate is linear on each segment.
    """

    curve_type: Literal["interpolated_linear_curve"] = "interpolated_linear_curve"

    def _zero_rate(self, tau: FloatArray) -> FloatArray:
        return np.interp(tau, self._ttm, self._rates)

    def _zero_rate_derivative(self, tau: FloatArray) -> FloatArray:
        t, r = self._ttm, self._rates
        if t.size < 2:
            return np.zeros_like(tau)
        slope = np.diff(r) / np.diff(t)
        idx = np.clip(np.searchsorted(t, tau, side="right") - 1, 0, slope.size - 1)
        inside = (tau >= t[0]) & (tau <= t[-1])
        return np.where(inside, slope[idx], 0.0)


class InterpolatedMonotonicCubicCurve(InterpolatedYieldCurve):
    r"""Yield curve interpolating the zero rate with a monotone cubic spline.

    The zero rate $r(\tau)$ is interpolated with a shape-preserving cubic Hermite
    spline (PCHIP, Fritsch-Carlson) that never introduces a new local maximum or
    minimum between two nodes, giving a smooth zero rate and forward rate.
    """

    curve_type: Literal["interpolated_monotonic_cubic_curve"] = (
        "interpolated_monotonic_cubic_curve"
    )

    def _zero_rate(self, tau: FloatArray) -> FloatArray:
        t, r = self._ttm, self._rates
        if t.size < 2:
            return np.full(np.shape(tau), r[0]) if r.size else np.zeros_like(tau)
        clamped = np.clip(tau, t[0], t[-1])
        return PchipInterpolator(t, r)(clamped)

    def _zero_rate_derivative(self, tau: FloatArray) -> FloatArray:
        t, r = self._ttm, self._rates
        if t.size < 2:
            return np.zeros_like(tau)
        clamped = np.clip(tau, t[0], t[-1])
        inside = (tau >= t[0]) & (tau <= t[-1])
        dr = PchipInterpolator(t, r).derivative()(clamped)
        return np.where(inside, dr, 0.0)


class InterpolatedYieldCurveCalibration(YieldCurveCalibration[InterpolatedYieldCurve]):
    """Calibration wrapper for an interpolated yield curve.

    The interpolated curve passes exactly through its nodes, so calibration is
    direct: the anchor dates and rates are set from the input times to maturity
    and continuously compounded rates. The free parameters are the anchor rates.
    """

    def get_params(self) -> FloatArray:
        return np.array([float(r) for r in self.yield_curve.anchor_rates], dtype=float)

    def set_params(self, params: FloatArray) -> None:
        curve = self.yield_curve
        rates = np.asarray(params, dtype=float)
        curve.anchor_rates = [Decimal(str(round(float(r), 10))) for r in rates]
        curve._rates = rates

    def get_bounds(self) -> Bounds:
        n = len(self.yield_curve.anchor_rates)
        return Bounds(np.full(n, -np.inf), np.full(n, np.inf))

    def calibrate(
        self,
        ttm: Annotated[ArrayLike, Doc("Times to maturity in years.")],
        rates: Annotated[
            ArrayLike, Doc("Continuously compounded rates, same length as ttm.")
        ],
    ) -> InterpolatedYieldCurve:
        """Set the curve nodes so it reprices the given rates exactly.

        Maturity dates are reconstructed from the times to maturity relative to
        [ref_date][quantflow.rates.yield_curve.YieldCurve.ref_date] on an
        ACT/365 basis.
        """
        ttm_ = np.asarray(ttm, dtype=float)
        rates_ = np.asarray(rates, dtype=float)
        order = np.argsort(ttm_)
        curve = self.yield_curve
        ref = curve.ref_date
        curve.anchor_dates = [
            ref + timedelta(seconds=float(t) * _YEAR) for t in ttm_[order]
        ]
        curve.anchor_rates = [Decimal(str(round(float(r), 10))) for r in rates_[order]]
        curve._ttm = ttm_[order]
        curve._rates = rates_[order]
        return curve
