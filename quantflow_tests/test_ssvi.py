from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from pydantic import ValidationError

from quantflow.options.ssvi import SSVI, VarianceCurve

TTM = 1.0
K = np.linspace(-0.5, 0.5, 21)


def make_ssvi(**kwargs: float) -> SSVI:
    defaults = dict(rho=-0.3, eta=1.0, gamma=0.5, theta=0.04)
    defaults.update(kwargs)
    theta = defaults.pop("theta")
    return SSVI(
        **{k: Decimal(str(v)) for k, v in defaults.items()},
        variance_curve=VarianceCurve(
            ttm=[Decimal(str(TTM))],
            theta=[Decimal(str(theta))],
        ),
    )


# ---------------------------------------------------------------------------
# phi
# ---------------------------------------------------------------------------


def test_phi_power_law() -> None:
    ssvi = make_ssvi(eta=1.0, gamma=0.5, theta=0.04)
    expected = 1.0 / (0.04**0.5 * 1.04**0.5)
    assert ssvi.phi(TTM) == pytest.approx(expected)


def test_phi_decreases_with_theta() -> None:
    phi_short = make_ssvi(theta=0.01).phi(TTM)
    phi_long = make_ssvi(theta=0.25).phi(TTM)
    assert phi_short > phi_long


# ---------------------------------------------------------------------------
# variance curve
# ---------------------------------------------------------------------------


def test_variance_curve_interpolates_nodes() -> None:
    curve = VarianceCurve(
        ttm=[Decimal("0.25"), Decimal("0.5"), Decimal("1.0")],
        theta=[Decimal("0.02"), Decimal("0.05"), Decimal("0.09")],
    )
    assert curve.total_variance(0.5) == pytest.approx(0.05)


def test_variance_curve_is_monotone_between_nodes() -> None:
    curve = VarianceCurve(
        ttm=[Decimal("0.25"), Decimal("0.5"), Decimal("1.0")],
        theta=[Decimal("0.02"), Decimal("0.05"), Decimal("0.09")],
    )
    grid = np.linspace(0.25, 1.0, 101)
    theta = np.asarray(curve.total_variance(grid), dtype=float)
    assert np.all(np.diff(theta) >= 0)
    assert np.all(np.asarray(curve.derivative(grid), dtype=float) >= -1e-12)


def test_variance_curve_rejects_decreasing_theta() -> None:
    with pytest.raises(ValidationError):
        VarianceCurve(
            ttm=[Decimal("0.25"), Decimal("0.5"), Decimal("1.0")],
            theta=[Decimal("0.02"), Decimal("0.05"), Decimal("0.04")],
        )


# ---------------------------------------------------------------------------
# total_variance
# ---------------------------------------------------------------------------


def test_total_variance_is_positive() -> None:
    ssvi = make_ssvi()
    w = ssvi.total_variance(K, TTM)
    assert np.all(w > 0)


def test_total_variance_atm_equals_theta() -> None:
    ssvi = make_ssvi()
    assert ssvi.total_variance(0.0, TTM) == pytest.approx(float(ssvi.theta))


def test_total_variance_symmetric_when_rho_zero() -> None:
    ssvi = make_ssvi(rho=0.0)
    w = np.asarray(ssvi.total_variance(K, TTM), dtype=float)
    assert np.allclose(w, w[::-1])


def test_total_variance_skew_sign() -> None:
    # negative rho => put wing (k < 0) above call wing (k > 0)
    ssvi = make_ssvi(rho=-0.5)
    w = np.asarray(ssvi.total_variance(K, TTM), dtype=float)
    assert w[0] > w[-1]


def test_total_variance_scalar_input() -> None:
    ssvi = make_ssvi()
    w = ssvi.total_variance(0.0, TTM)
    assert isinstance(w, float)


# ---------------------------------------------------------------------------
# iv
# ---------------------------------------------------------------------------


def test_iv_positive() -> None:
    ssvi = make_ssvi()
    iv = ssvi.iv(K, TTM)
    assert np.all(iv > 0)


def test_iv_consistent_with_total_variance() -> None:
    ssvi = make_ssvi()
    iv = ssvi.iv(K, TTM)
    w = ssvi.total_variance(K, TTM)
    assert np.allclose(iv**2 * TTM, w)


def test_iv_scales_with_ttm() -> None:
    ssvi = make_ssvi()
    iv1 = ssvi.iv(K, 1.0)
    iv2 = ssvi.iv(K, 0.25)
    # same total variance, shorter ttm => higher iv
    assert np.all(iv2 > iv1)


# ---------------------------------------------------------------------------
# no_butterfly_arbitrage
# ---------------------------------------------------------------------------


def test_no_butterfly_arbitrage_typical_params() -> None:
    assert make_ssvi().no_butterfly_arbitrage()


def test_butterfly_arbitrage_violated_for_large_eta() -> None:
    assert not make_ssvi(eta=15.0, theta=1.0).no_butterfly_arbitrage()


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_recovers_parameters() -> None:
    true_ssvi = make_ssvi(rho=-0.2, eta=0.8, gamma=0.5, theta=0.05)
    iv_obs = true_ssvi.iv(K, TTM)
    fitted = SSVI.fit(K, iv_obs, TTM)
    assert float(fitted.rho) == pytest.approx(float(true_ssvi.rho), abs=1e-4)
    assert float(fitted.eta) == pytest.approx(float(true_ssvi.eta), abs=1e-4)
    assert float(fitted.theta) == pytest.approx(float(true_ssvi.theta), abs=1e-4)


def test_fit_reproduces_ivs() -> None:
    true_ssvi = make_ssvi()
    iv_obs = true_ssvi.iv(K, TTM)
    fitted = SSVI.fit(K, iv_obs, TTM)
    iv_fit = fitted.iv(K, TTM)
    assert np.allclose(iv_fit, iv_obs, atol=1e-5)


def test_fit_respects_bounds() -> None:
    true_ssvi = make_ssvi()
    iv_obs = true_ssvi.iv(K, TTM)
    fitted = SSVI.fit(K, iv_obs, TTM)
    assert -1 < float(fitted.rho) < 1
    assert float(fitted.eta) > 0
    assert float(fitted.theta) > 0


def test_fit_fixed_gamma() -> None:
    true_ssvi = make_ssvi()
    iv_obs = true_ssvi.iv(K, TTM)
    fitted = SSVI.fit(K, iv_obs, TTM, gamma=0.4)
    assert float(fitted.gamma) == pytest.approx(0.4)


def test_fit_skewed_smile() -> None:
    true_ssvi = make_ssvi(rho=-0.6)
    iv_obs = true_ssvi.iv(K, TTM)
    fitted = SSVI.fit(K, iv_obs, TTM)
    iv_fit = fitted.iv(K, TTM)
    assert np.allclose(iv_fit, iv_obs, atol=1e-5)


# ---------------------------------------------------------------------------
# fit_surface
# ---------------------------------------------------------------------------


def _synthetic_surface(
    rho: float, eta: float, gamma: float, thetas: list[float], ttms: list[float]
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    slices = []
    for theta, ttm in zip(thetas, ttms):
        ssvi = make_ssvi(rho=rho, eta=eta, gamma=gamma, theta=theta)
        slices.append((K, np.asarray(ssvi.iv(K, ttm), dtype=float), ttm))
    return slices


def test_fit_surface_recovers_parameters() -> None:
    thetas = [0.02, 0.05, 0.09]
    ttms = [0.25, 0.5, 1.0]
    slices = _synthetic_surface(-0.3, 0.9, 0.45, thetas, ttms)
    fitted = SSVI.fit_surface(slices)
    assert float(fitted.rho) == pytest.approx(-0.3, abs=1e-3)
    assert float(fitted.eta) == pytest.approx(0.9, abs=1e-2)
    assert float(fitted.gamma) == pytest.approx(0.45, abs=1e-2)
    for ttm, theta in zip(ttms, thetas):
        assert fitted.variance_curve.total_variance(ttm) == pytest.approx(
            theta, abs=1e-4
        )


def test_fit_surface_shares_global_parameters() -> None:
    slices = _synthetic_surface(-0.2, 1.1, 0.5, [0.03, 0.06], [0.5, 1.0])
    fitted = SSVI.fit_surface(slices)
    assert len(set(fitted.variance_curve.theta)) == 2
    assert float(fitted.rho) == pytest.approx(-0.2, abs=1e-3)
    assert float(fitted.eta) == pytest.approx(1.1, abs=1e-2)
    assert float(fitted.gamma) == pytest.approx(0.5, abs=1e-2)


def test_fit_surface_reproduces_ivs() -> None:
    slices = _synthetic_surface(-0.4, 0.8, 0.5, [0.02, 0.05, 0.09], [0.25, 0.5, 1.0])
    fitted = SSVI.fit_surface(slices)
    for k, iv_obs, ttm in slices:
        assert np.allclose(fitted.iv(k, ttm), iv_obs, atol=1e-5)


def test_fit_surface_thetas_increasing() -> None:
    # increasing ATM total variance => no calendar spread arbitrage
    slices = _synthetic_surface(-0.3, 0.9, 0.5, [0.02, 0.05, 0.09], [0.25, 0.5, 1.0])
    fitted = SSVI.fit_surface(slices)
    thetas = [float(theta) for theta in fitted.variance_curve.theta]
    assert thetas == sorted(thetas)
    assert fitted.no_static_arbitrage()


def test_fit_surface_enforces_monotone_variance_curve() -> None:
    slices = _synthetic_surface(-0.3, 0.9, 0.5, [0.02, 0.015, 0.09], [0.25, 0.5, 1.0])
    fitted = SSVI.fit_surface(slices)
    thetas = [float(theta) for theta in fitted.variance_curve.theta]
    assert thetas == sorted(thetas)
