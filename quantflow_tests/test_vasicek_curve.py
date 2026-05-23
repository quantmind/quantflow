from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from quantflow.rates.vasicek import VasicekCurve


def test_vasicek_process_mapping() -> None:
    curve = VasicekCurve(
        rate=Decimal("0.03"),
        kappa=Decimal("1.2"),
        theta=Decimal("0.04"),
        sigma=Decimal("0.1"),
    )
    process = curve.process()
    assert process.rate == pytest.approx(0.03)
    assert process.kappa == pytest.approx(1.2)
    assert process.theta == pytest.approx(0.04)
    assert process.bdlp.sigma == pytest.approx(0.1)


def test_vasicek_forward_and_discount_shapes() -> None:
    curve = VasicekCurve(
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


def test_vasicek_calibrate_recovers_curve() -> None:
    true_curve = VasicekCurve(
        rate=Decimal("0.03"),
        kappa=Decimal("1.5"),
        theta=Decimal("0.04"),
        sigma=Decimal("0.06"),
    )
    ttm = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0], dtype=float)
    rates = -np.log(np.asarray(true_curve.discount_factor(ttm))) / ttm
    fitted = VasicekCurve.calibrate(ttm, rates)
    fitted_df = np.asarray(fitted.discount_factor(ttm))
    true_df = np.asarray(true_curve.discount_factor(ttm))
    np.testing.assert_allclose(fitted_df, true_df, atol=3e-3)
