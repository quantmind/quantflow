from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from quantflow.rates.cir import CIRCurve


def test_cir_process_mapping() -> None:
    curve = CIRCurve(
        rate=Decimal("0.03"),
        kappa=Decimal("1.2"),
        theta=Decimal("0.04"),
        sigma=Decimal("0.1"),
    )
    process = curve.process()
    assert process.rate == pytest.approx(0.03)
    assert process.kappa == pytest.approx(1.2)
    assert process.theta == pytest.approx(0.04)
    assert process.sigma == pytest.approx(0.1)


def test_cir_forward_and_discount_shapes() -> None:
    curve = CIRCurve(
        rate=Decimal("0.03"),
        kappa=Decimal("0.8"),
        theta=Decimal("0.05"),
        sigma=Decimal("0.1"),
    )
    ttms = np.array([0.0, 0.5, 1.0, 2.0])
    fwd = np.asarray(curve.instantaneous_forward_rate(ttms))
    df = np.asarray(curve.discount_factor(ttms))
    assert fwd.shape == ttms.shape
    assert df.shape == ttms.shape
    assert df[0] == pytest.approx(1.0)
    assert df[-1] < df[1]


def test_cir_forward_rate_at_zero() -> None:
    curve = CIRCurve(
        rate=Decimal("0.05"),
        kappa=Decimal("1.0"),
        theta=Decimal("0.04"),
        sigma=Decimal("0.2"),
    )
    fwd = curve.instantaneous_forward_rate(0.0)
    assert fwd == pytest.approx(0.05, rel=1e-6)


def test_cir_calibrate_recovers_curve() -> None:
    true_curve = CIRCurve(
        rate=Decimal("0.03"),
        kappa=Decimal("1.5"),
        theta=Decimal("0.04"),
        sigma=Decimal("0.06"),
    )
    ttm = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0], dtype=float)
    rates = -np.log(np.asarray(true_curve.discount_factor(ttm))) / ttm
    fitted = CIRCurve().calibrator().calibrate(ttm, rates)
    fitted_df = np.asarray(fitted.discount_factor(ttm))
    true_df = np.asarray(true_curve.discount_factor(ttm))
    np.testing.assert_allclose(fitted_df, true_df, atol=3e-3)


def test_cir_calibrate_feller_condition() -> None:
    """Calibration must always produce parameters satisfying the Feller condition."""
    ttm = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    rates = np.array([0.05, 0.048, 0.045, 0.042, 0.040, 0.039], dtype=float)
    fitted = CIRCurve().calibrator().calibrate(ttm, rates)
    process = fitted.process()
    assert process.feller_condition >= 0.0
    assert process.is_positive is True


def test_cir_continuously_compounded_rate() -> None:
    curve = CIRCurve(
        rate=Decimal("0.04"),
        kappa=Decimal("1.0"),
        theta=Decimal("0.05"),
        sigma=Decimal("0.15"),
    )
    ttm = np.array([1.0, 5.0, 10.0])
    df = np.asarray(curve.discount_factor(ttm))
    expected_rates = -np.log(df) / ttm
    computed_rates = np.asarray(curve.continuously_compounded_rate(ttm))
    np.testing.assert_allclose(computed_rates, expected_rates, rtol=1e-10)
