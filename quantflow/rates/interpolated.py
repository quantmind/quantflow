from __future__ import annotations

import enum
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


class InterpolationType(enum.StrEnum):
    """Interpolation method for the log discount factor"""

    LINEAR = enum.auto()
    """Piecewise linear in the log discount factor (piecewise constant forwards)."""

    MONOTONE_CUBIC = enum.auto()
    """Shape-preserving cubic Hermite spline (PCHIP, Fritsch-Carlson) that never
    introduces a new local maximum or minimum between two nodes."""


class InterpolatedYieldCurve(YieldCurve, arbitrary_types_allowed=True):
    r"""Yield curve built by interpolating the log discount factor on a set of nodes.

    The curve is defined by continuously compounded zero rates $r_i$ at a set of
    anchor dates. Times to maturity $\tau_i$ are measured from
    [ref_date][..ref_date] on an ACT/365 basis, and the log discount factor at
    each node is $g_i = -r_i \tau_i$. The curve interpolates $g(\tau) = \ln
    D(\tau)$ between the nodes, which keeps the instantaneous forward rate
    $f(\tau) = -g'(\tau)$ simple: piecewise constant for linear interpolation and
    smooth for cubic interpolation.

    The node $\tau = 0$ with $g = 0$ (i.e. $D(0) = 1$) is added automatically.
    Beyond the last node the instantaneous forward rate is held flat (constant
    forward extrapolation).
    """

    curve_type: Literal["interpolated_yield_curve"] = "interpolated_yield_curve"
    anchor_dates: list[datetime] = Field(
        description="Maturity dates of the interpolation nodes, strictly after the "
        "reference date and in increasing order"
    )
    anchor_rates: list[DecimalNumber] = Field(
        description="Continuously compounded zero rates at each anchor date "
        "(0.05 means 5%), same length as anchor_dates"
    )
    interpolation_type: InterpolationType = Field(
        default=InterpolationType.LINEAR,
        description="Interpolation method for the log discount factor: "
        "LINEAR or MONOTONE_CUBIC (PCHIP)",
    )

    _ttm: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))
    _log_discount: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))

    def model_post_init(self, context: object) -> None:
        """Cache the times to maturity and log discount factors at the nodes."""
        if len(self.anchor_dates) != len(self.anchor_rates):
            raise ValueError("anchor_dates and anchor_rates must have equal length")
        if not self.anchor_dates:
            raise ValueError("at least one anchor is required")
        ttm = self._year_fractions()
        if np.any(ttm <= 0):
            raise ValueError("anchor_dates must be strictly after the reference date")
        if np.any(np.diff(ttm) <= 0):
            raise ValueError("anchor_dates must be strictly increasing")
        rates = np.array([float(r) for r in self.anchor_rates], dtype=float)
        self._ttm = ttm
        self._log_discount = -rates * ttm

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

    def _nodes(self) -> tuple[FloatArray, FloatArray]:
        """Node times and log discount factors, with the origin pinned to (0, 0)."""
        t = np.concatenate([[0.0], self._ttm])
        g = np.concatenate([[0.0], self._log_discount])
        return t, g

    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        t, g = self._nodes()
        tau = np.maximum(np.asarray(ttm, dtype=float), 0.0)
        tmax = t[-1]
        if self.interpolation_type is InterpolationType.LINEAR:
            slope = np.diff(g) / np.diff(t)
            idx = np.clip(
                np.searchsorted(t, np.minimum(tau, tmax), side="right") - 1,
                0,
                slope.size - 1,
            )
            f = -slope[idx]
        else:
            dg = PchipInterpolator(t, g).derivative()
            f = -dg(np.minimum(tau, tmax))
        return maybe_float(f)

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        t, g = self._nodes()
        tau = np.maximum(np.asarray(ttm, dtype=float), 0.0)
        tmax = t[-1]
        inside = np.minimum(tau, tmax)
        if self.interpolation_type is InterpolationType.LINEAR:
            slope = np.diff(g) / np.diff(t)
            gi = np.interp(inside, t, g)
            f_last = -slope[-1]
        else:
            pch = PchipInterpolator(t, g)
            gi = pch(inside)
            f_last = -float(pch.derivative()(tmax))
        # constant forward rate extrapolation beyond the last node
        gi = np.where(tau > tmax, g[-1] - f_last * (tau - tmax), gi)
        return maybe_float(np.exp(gi))


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
        curve._log_discount = -rates * curve._ttm

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
        [ref_date][..ref_date] on an ACT/365 basis.
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
        curve._log_discount = -rates_[order] * ttm_[order]
        return curve
