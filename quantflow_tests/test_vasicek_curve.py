from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quantflow.rates.calibration import tenor_to_years
from quantflow.rates.no_discount import NoDiscountCurve
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
    fitted = VasicekCurve().calibrator().calibrate(ttm, rates)
    fitted_df = np.asarray(fitted.discount_factor(ttm))
    true_df = np.asarray(true_curve.discount_factor(ttm))
    np.testing.assert_allclose(fitted_df, true_df, atol=3e-3)


def _simulate_panel(
    curve: VasicekCurve,
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
    q_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    r = float(curve.rate)
    short_rates = np.empty(n_obs)
    for t in range(n_obs):
        short_rates[t] = r
        r = theta * (1.0 - phi) + phi * r + q_std * rng.standard_normal()
    ttm = np.array([tenor_to_years(s) for s in tenors], dtype=float)
    rows = []
    for r_t in short_rates:
        sample = VasicekCurve(
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


def test_vasicek_calibrate_historical_rates_recovers_parameters() -> None:
    true_curve = VasicekCurve(
        rate=Decimal("0.03"),
        kappa=Decimal("0.6"),
        theta=Decimal("0.04"),
        sigma=Decimal("0.012"),
    )
    tenors = ["3m", "6m", "1y", "2y", "5y", "10y"]
    panel = _simulate_panel(
        true_curve, tenors, n_obs=600, dt=1 / 52, noise=1e-5, seed=42
    )
    fitted = VasicekCurve().calibrator().calibrate_historical_rates_dataframe(panel)
    assert float(fitted.kappa) == pytest.approx(0.6, rel=0.4)
    assert float(fitted.theta) == pytest.approx(0.04, abs=0.01)
    assert float(fitted.sigma) == pytest.approx(0.012, rel=0.4)


def test_vasicek_historical_rates_compounding_frequency() -> None:
    true_curve = VasicekCurve(
        rate=Decimal("0.025"),
        kappa=Decimal("0.5"),
        theta=Decimal("0.035"),
        sigma=Decimal("0.01"),
    )
    tenors = ["6m", "1y", "2y", "5y"]
    panel_cc = _simulate_panel(
        true_curve, tenors, n_obs=400, dt=1 / 52, noise=1e-6, seed=7
    )
    # round-trip continuous -> annual -> continuous gives same answer
    panel_annual = np.expm1(panel_cc)
    fitted_cc = (
        VasicekCurve().calibrator().calibrate_historical_rates_dataframe(panel_cc)
    )
    fitted_annual = (
        VasicekCurve()
        .calibrator()
        .calibrate_historical_rates_dataframe(panel_annual, frequency=1)
    )
    assert float(fitted_cc.theta) == pytest.approx(float(fitted_annual.theta), abs=1e-4)
    assert float(fitted_cc.kappa) == pytest.approx(float(fitted_annual.kappa), rel=1e-3)


def test_calibrate_historical_rates_not_implemented_on_no_discount() -> None:
    tenors = ["6m", "1y"]
    panel = pd.DataFrame(
        np.zeros((10, 2)),
        index=pd.date_range("2020-01-01", periods=10, freq="W"),
        columns=tenors,
    )
    with pytest.raises(NotImplementedError):
        NoDiscountCurve().calibrator().calibrate_historical_rates_dataframe(panel)
