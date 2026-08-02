from __future__ import annotations

from typing import Literal

import numpy as np

from quantflow.utils.types import FloatArrayLike

from .yield_curve import YieldCurve


class NoDiscountCurve(YieldCurve):
    """Flat yield curve with zero rates (discount factor is always 1).

    It has no parameters to calibrate, therefore
    [calibrator][quantflow.rates.yield_curve.YieldCurve.calibrator] returns
    None and the curve is treated as fixed by curve calibrations.
    """

    curve_type: Literal["no_discount_curve"] = "no_discount_curve"

    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        arr = np.asarray(ttm, dtype=float)
        return np.zeros_like(arr) if arr.ndim > 0 else 0.0

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        arr = np.asarray(ttm, dtype=float)
        return np.ones_like(arr) if arr.ndim > 0 else 1.0
