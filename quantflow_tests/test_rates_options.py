from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.optimize import Bounds

from quantflow.rates.options import OptionsDiscountingCalibration, YieldCurveCalibration
from quantflow.rates.yield_curve import YieldCurve
from quantflow.utils.types import FloatArray, FloatArrayLike, maybe_float


class ExponentialCurve(YieldCurve):
    curve_type: Literal["exp_curve"] = "exp_curve"
    rate: float = 0.05

    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        arr = np.asarray(ttm, dtype=float)
        result = np.full_like(arr, self.rate)
        return maybe_float(result)

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        arr = np.asarray(ttm, dtype=float)
        result = np.exp(-self.rate * arr)
        return maybe_float(result)

    def jacobian(self, ttm: FloatArrayLike) -> FloatArray | None:
        arr = np.asarray(ttm, dtype=float)
        return (-arr * np.exp(-self.rate * arr)).reshape(-1, 1)

    @classmethod
    def calibrate(cls, ttm: ArrayLike, rates: ArrayLike) -> "ExponentialCurve":
        return cls(rate=float(np.mean(np.asarray(rates, dtype=float))))


class ExponentialCurveCalibration(YieldCurveCalibration[ExponentialCurve]):
    def get_params(self) -> FloatArray:
        return np.array([self.yield_curve.rate], dtype=float)

    def set_params(self, params: FloatArray) -> None:
        self.yield_curve.rate = float(params[0])

    def get_bounds(self) -> Bounds:
        return Bounds([0.0], [1.0])


def _ttm_strikes_cp(
    asset_rate: float, quote_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ttm = np.array([0.5, 1.0, 2.0, 3.0], dtype=float)
    strikes = np.array([0.9, 1.0, 1.1, 1.2], dtype=float)
    da = np.exp(-asset_rate * ttm)
    dq = np.exp(-quote_rate * ttm)
    cp = da - dq * strikes
    return ttm, strikes, cp


def test_yield_curve_calibration_base_calibrate() -> None:
    ttm = np.array([0.5, 1.0, 2.0, 3.0], dtype=float)
    true_rate = 0.03
    target = np.exp(-true_rate * ttm)
    cal = ExponentialCurveCalibration(yield_curve=ExponentialCurve(rate=0.10))
    fitted = cal.calibrate(ttm, target)
    assert fitted.rate == pytest.approx(true_rate, abs=1e-4)


def test_options_discounting_joint_calibration() -> None:
    ttm, strikes, cp = _ttm_strikes_cp(asset_rate=0.02, quote_rate=0.04)
    asset_cal = ExponentialCurveCalibration(yield_curve=ExponentialCurve(rate=0.08))
    quote_cal = ExponentialCurveCalibration(yield_curve=ExponentialCurve(rate=0.09))
    calibration = OptionsDiscountingCalibration(
        asset_curve=asset_cal,
        quote_curve=quote_cal,
        cp=cp,
        strikes=strikes,
        ttm=ttm,
    )
    asset_curve, quote_curve = calibration.calibrate()
    assert asset_curve.rate == pytest.approx(0.02, abs=1e-3)
    assert quote_curve.rate == pytest.approx(0.04, abs=1e-3)


def test_options_discounting_asset_only_calibration() -> None:
    ttm, strikes, cp = _ttm_strikes_cp(asset_rate=0.02, quote_rate=0.04)
    asset_cal = ExponentialCurveCalibration(yield_curve=ExponentialCurve(rate=0.12))
    fixed_quote = ExponentialCurve(rate=0.04)
    calibration = OptionsDiscountingCalibration(
        asset_curve=asset_cal,
        quote_curve=fixed_quote,
        cp=cp,
        strikes=strikes,
        ttm=ttm,
    )
    asset_curve, quote_curve = calibration.calibrate()
    assert asset_curve.rate == pytest.approx(0.02, abs=1e-3)
    assert quote_curve.rate == pytest.approx(0.04, abs=1e-9)


def test_options_discounting_quote_only_calibration() -> None:
    ttm, strikes, cp = _ttm_strikes_cp(asset_rate=0.02, quote_rate=0.04)
    fixed_asset = ExponentialCurve(rate=0.02)
    quote_cal = ExponentialCurveCalibration(yield_curve=ExponentialCurve(rate=0.10))
    calibration = OptionsDiscountingCalibration(
        asset_curve=fixed_asset,
        quote_curve=quote_cal,
        cp=cp,
        strikes=strikes,
        ttm=ttm,
    )
    asset_curve, quote_curve = calibration.calibrate()
    assert asset_curve.rate == pytest.approx(0.02, abs=1e-9)
    assert quote_curve.rate == pytest.approx(0.04, abs=1e-3)


def test_options_discounting_both_fixed_returns_inputs() -> None:
    ttm, strikes, cp = _ttm_strikes_cp(asset_rate=0.02, quote_rate=0.04)
    fixed_asset = ExponentialCurve(rate=0.02)
    fixed_quote = ExponentialCurve(rate=0.04)
    calibration = OptionsDiscountingCalibration(
        asset_curve=fixed_asset,
        quote_curve=fixed_quote,
        cp=cp,
        strikes=strikes,
        ttm=ttm,
    )
    asset_curve, quote_curve = calibration.calibrate()
    assert asset_curve is fixed_asset
    assert quote_curve is fixed_quote
