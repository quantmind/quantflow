from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import Bounds
from typing_extensions import Annotated, Doc

from quantflow.utils.types import FloatArray, FloatArrayLike

from .options import YieldCurveCalibration
from .yield_curve import YieldCurve


class NoDiscount(YieldCurve):
    """Flat yield curve with zero rates (discount factor is always 1)."""

    curve_type: Literal["no_discount"] = "no_discount"

    def calibrator(self) -> NoDiscountCalibration:
        """Return a [NoDiscountCalibration][.NoDiscountCalibration] wrapping
        this curve."""
        return NoDiscountCalibration(yield_curve=self)

    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        arr = np.asarray(ttm, dtype=float)
        return np.zeros_like(arr) if arr.ndim > 0 else 0.0

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        arr = np.asarray(ttm, dtype=float)
        return np.ones_like(arr) if arr.ndim > 0 else 1.0


class NoDiscountCalibration(YieldCurveCalibration[NoDiscount]):
    """No-op calibration wrapper for NoDiscount (no parameters to fit)."""

    def get_params(self) -> FloatArray:
        return np.array([], dtype=float)

    def set_params(self, params: FloatArray) -> None:
        pass

    def get_bounds(self) -> Bounds:
        return Bounds([], [])

    def calibrate(
        self,
        ttm: Annotated[ArrayLike, Doc("Times to maturity in years.")],
        rates: Annotated[
            ArrayLike, Doc("Continuously compounded rates, same length as ttm.")
        ],
    ) -> NoDiscount:
        """No-op: NoDiscount has no parameters to calibrate."""
        return self.yield_curve
