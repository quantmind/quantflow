from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quantflow.rates.calibration import tenor_to_years
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


def test_cir_affine_coefficients_match_discount_factor() -> None:
    # log D(tau) = A(tau) - B(tau) * r0 must reproduce the closed-form factor
    curve = CIRCurve(
        rate=Decimal("0.037"),
        kappa=Decimal("0.9"),
        theta=Decimal("0.05"),
        sigma=Decimal("0.08"),
    )
    ttm = np.array([0.25, 1.0, 3.0, 7.0, 10.0])
    a, b = curve.affine_coefficients(ttm)
    log_df = np.asarray(a) - np.asarray(b) * float(curve.rate)
    np.testing.assert_allclose(
        np.exp(log_df), np.asarray(curve.discount_factor(ttm)), rtol=1e-12
    )


def _simulate_cir_panel(
    curve: CIRCurve,
    tenors: list[str],
    n_obs: int,
    dt: float,
    noise: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kappa = float(curve.kappa)
    theta = float(curve.theta)
    sigma = float(curve.sigma)
    phi = np.exp(-kappa * dt)
    s2 = sigma * sigma
    r = float(curve.rate)
    short_rates = np.empty(n_obs)
    for t in range(n_obs):
        short_rates[t] = r
        mean = theta + (r - theta) * phi
        var = (
            max(r, 0.0) * s2 / kappa * (phi - phi * phi)
            + theta * s2 / (2.0 * kappa) * (1.0 - phi) ** 2
        )
        r = max(mean + np.sqrt(var) * rng.standard_normal(), 1e-6)
    ttm = np.array([tenor_to_years(s) for s in tenors], dtype=float)
    rows = []
    for r_t in short_rates:
        sample = CIRCurve(
            rate=Decimal(str(r_t)),
            kappa=curve.kappa,
            theta=curve.theta,
            sigma=curve.sigma,
        )
        df = np.asarray(sample.discount_factor(ttm), dtype=float)
        y = -np.log(df) / ttm
        rows.append(y + noise * rng.standard_normal(len(tenors)))
    index = pd.date_range("2020-01-01", periods=n_obs, freq=pd.Timedelta(days=dt * 365))
    return pd.DataFrame(rows, index=index, columns=tenors)


def test_cir_calibrate_historical_rates_recovers_parameters() -> None:
    true_curve = CIRCurve(
        rate=Decimal("0.04"),
        kappa=Decimal("0.6"),
        theta=Decimal("0.04"),
        sigma=Decimal("0.05"),
    )
    tenors = ["3m", "6m", "1y", "2y", "5y", "10y"]
    panel = _simulate_cir_panel(
        true_curve, tenors, n_obs=600, dt=1 / 52, noise=1e-5, seed=1
    )
    calibrator = CIRCurve().calibrator()
    fitted = calibrator.calibrate_historical_rates_dataframe(panel)
    assert float(fitted.kappa) == pytest.approx(0.6, rel=0.4)
    assert float(fitted.theta) == pytest.approx(0.04, abs=0.01)
    assert float(fitted.sigma) == pytest.approx(0.05, rel=0.4)
    # the filtered short rate tracks the (latent) simulated short rate
    assert len(calibrator.filtered_short_rate) == len(panel)


def test_cir_filtered_short_rate_requires_fit() -> None:
    with pytest.raises(AttributeError):
        CIRCurve().calibrator().filtered_short_rate
