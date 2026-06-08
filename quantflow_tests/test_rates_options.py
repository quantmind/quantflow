from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from quantflow.rates.calibration import OptionsDiscountingCalibration
from quantflow.rates.nelson_siegel import NelsonSiegelCurve


def _ttm_strikes_cp(
    asset_rate: float, quote_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ttm = np.array([0.5, 1.0, 2.0, 3.0], dtype=float)
    strikes = np.array([0.9, 1.0, 1.1, 1.2], dtype=float)
    da = np.exp(-asset_rate * ttm)
    dq = np.exp(-quote_rate * ttm)
    cp = da - dq * strikes
    return ttm, strikes, cp


def _flat(rate: float) -> NelsonSiegelCurve:
    return NelsonSiegelCurve(beta1=Decimal(str(rate)))


def test_calibrate_flat_rate() -> None:
    ttm = np.array([0.5, 1.0, 2.0, 3.0], dtype=float)
    true_rate = 0.03
    rates = np.full_like(ttm, true_rate)
    fitted = NelsonSiegelCurve().calibrator().calibrate(ttm, rates)
    assert float(fitted.beta1) == pytest.approx(true_rate, abs=1e-4)


def test_options_discounting_joint_calibration() -> None:
    ttm, strikes, cp = _ttm_strikes_cp(asset_rate=0.02, quote_rate=0.04)
    calibration = OptionsDiscountingCalibration(
        asset_curve=NelsonSiegelCurve().calibrator(),
        quote_curve=NelsonSiegelCurve().calibrator(),
        cp=cp,
        strikes=strikes,
        ttm=ttm,
    )
    asset_curve, quote_curve = calibration.calibrate()
    assert isinstance(asset_curve, NelsonSiegelCurve)
    assert isinstance(quote_curve, NelsonSiegelCurve)
    da = np.asarray(asset_curve.discount_factor(ttm), dtype=float)
    dq = np.asarray(quote_curve.discount_factor(ttm), dtype=float)
    np.testing.assert_allclose(da - dq * strikes, cp, atol=1e-6)


def test_options_discounting_asset_only_calibration() -> None:
    ttm, strikes, cp = _ttm_strikes_cp(asset_rate=0.02, quote_rate=0.04)
    calibration = OptionsDiscountingCalibration(
        asset_curve=NelsonSiegelCurve().calibrator(),
        quote_curve=_flat(0.04),
        cp=cp,
        strikes=strikes,
        ttm=ttm,
    )
    asset_curve, quote_curve = calibration.calibrate()
    assert isinstance(asset_curve, NelsonSiegelCurve)
    assert isinstance(quote_curve, NelsonSiegelCurve)
    assert float(asset_curve.beta1) == pytest.approx(0.02, abs=1e-3)
    assert float(quote_curve.beta1) == pytest.approx(0.04, abs=1e-9)


def test_options_discounting_quote_only_calibration() -> None:
    ttm, strikes, cp = _ttm_strikes_cp(asset_rate=0.02, quote_rate=0.04)
    calibration = OptionsDiscountingCalibration(
        asset_curve=_flat(0.02),
        quote_curve=NelsonSiegelCurve().calibrator(),
        cp=cp,
        strikes=strikes,
        ttm=ttm,
    )
    asset_curve, quote_curve = calibration.calibrate()
    assert isinstance(asset_curve, NelsonSiegelCurve)
    assert isinstance(quote_curve, NelsonSiegelCurve)
    assert float(asset_curve.beta1) == pytest.approx(0.02, abs=1e-9)
    assert float(quote_curve.beta1) == pytest.approx(0.04, abs=1e-3)


def test_options_discounting_both_fixed_returns_inputs() -> None:
    ttm, strikes, cp = _ttm_strikes_cp(asset_rate=0.02, quote_rate=0.04)
    fixed_asset = _flat(0.02)
    fixed_quote = _flat(0.04)
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
