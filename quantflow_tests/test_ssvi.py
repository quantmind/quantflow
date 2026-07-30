from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from pydantic import ValidationError

from quantflow.options.ssvi import SSVI

TTM = 1.0
K = np.linspace(-0.5, 0.5, 21)


def make_ssvi(**kwargs: float) -> SSVI:
    defaults = dict(rho=-0.3, psi=0.2, theta=0.04)
    defaults.update(kwargs)
    return SSVI(
        ttm=[Decimal(str(TTM))],
        theta=[Decimal(str(defaults["theta"]))],
        psi=[Decimal(str(defaults["psi"]))],
        rho=[Decimal(str(defaults["rho"]))],
    )


def make_essvi(rhos: list[str] | None = None, psis: list[str] | None = None) -> SSVI:
    rhos = rhos if rhos is not None else ["-0.3", "-0.3", "-0.3"]
    psis = psis if psis is not None else ["0.12", "0.2", "0.26"]
    return SSVI(
        ttm=[Decimal("0.25"), Decimal("0.5"), Decimal("1.0")],
        theta=[Decimal("0.02"), Decimal("0.05"), Decimal("0.09")],
        psi=[Decimal(psi) for psi in psis],
        rho=[Decimal(rho) for rho in rhos],
    )


# ---------------------------------------------------------------------------
# node validation
# ---------------------------------------------------------------------------


def test_rejects_decreasing_theta() -> None:
    with pytest.raises(ValidationError):
        SSVI(
            ttm=[Decimal("0.25"), Decimal("0.5"), Decimal("1.0")],
            theta=[Decimal("0.02"), Decimal("0.05"), Decimal("0.04")],
            psi=[Decimal("0.12"), Decimal("0.2"), Decimal("0.26")],
            rho=[Decimal("-0.3"), Decimal("-0.3"), Decimal("-0.3")],
        )


def test_rejects_decreasing_psi() -> None:
    with pytest.raises(ValidationError):
        make_essvi(psis=["0.2", "0.15", "0.26"])


def test_rejects_rho_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        make_essvi(["-0.3", "-1.0", "-0.3"])


def test_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError):
        SSVI(
            ttm=[Decimal("0.25"), Decimal("0.5")],
            theta=[Decimal("0.02"), Decimal("0.05")],
            psi=[Decimal("0.12"), Decimal("0.2")],
            rho=[Decimal("-0.3")],
        )


# ---------------------------------------------------------------------------
# interpolation
# ---------------------------------------------------------------------------


def test_phi_is_curvature_over_theta() -> None:
    ssvi = make_ssvi(psi=0.2, theta=0.04)
    assert ssvi.phi(TTM) == pytest.approx(0.2 / 0.04)


def test_atm_variance_interpolates_nodes() -> None:
    assert make_essvi().atm_variance(0.5) == pytest.approx(0.05)


def test_atm_variance_interpolates_linearly() -> None:
    ssvi = make_essvi()
    assert ssvi.atm_variance(0.75) == pytest.approx(0.07)
    assert ssvi.curvature(0.75) == pytest.approx(0.23)


def test_atm_variance_is_monotone_between_nodes() -> None:
    ssvi = make_essvi()
    grid = np.linspace(0.25, 1.0, 101)
    theta = np.asarray(ssvi.atm_variance(grid), dtype=float)
    assert np.all(np.diff(theta) >= 0)


def test_atm_variance_extrapolates_flat() -> None:
    ssvi = make_essvi()
    assert ssvi.atm_variance(0.1) == pytest.approx(0.02)
    assert ssvi.atm_variance(2.0) == pytest.approx(0.09)


def test_correlation_interpolation() -> None:
    ssvi = make_essvi(["-0.5", "-0.3", "-0.1"])
    assert ssvi.correlation(0.5) == pytest.approx(-0.3)
    # linear in rho * psi and psi, so the ratio at mid points
    rho_psi = 0.5 * (-0.3 * 0.2) + 0.5 * (-0.1 * 0.26)
    psi = 0.5 * 0.2 + 0.5 * 0.26
    assert ssvi.correlation(0.75) == pytest.approx(rho_psi / psi)
    grid = np.linspace(0.25, 1.0, 101)
    rho = np.asarray(ssvi.correlation(grid), dtype=float)
    assert np.all(np.abs(rho) < 1)
    # flat extrapolation outside the node range
    assert ssvi.correlation(0.1) == pytest.approx(-0.5)
    assert ssvi.correlation(2.0) == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# total_variance
# ---------------------------------------------------------------------------


def test_total_variance_is_positive() -> None:
    ssvi = make_ssvi()
    w = ssvi.total_variance(K, TTM)
    assert np.all(w > 0)


def test_total_variance_atm_equals_theta() -> None:
    ssvi = make_ssvi()
    assert ssvi.total_variance(0.0, TTM) == pytest.approx(0.04)


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


def test_butterfly_arbitrage_violated_for_large_psi() -> None:
    assert not make_ssvi(psi=3.0, theta=1.0).no_butterfly_arbitrage()


# ---------------------------------------------------------------------------
# no_calendar_arbitrage
# ---------------------------------------------------------------------------


def test_no_calendar_arbitrage_constant_rho() -> None:
    assert make_essvi().no_calendar_arbitrage()


def test_no_calendar_arbitrage_time_varying_rho() -> None:
    assert make_essvi(["-0.4", "-0.35", "-0.2"]).no_calendar_arbitrage()


def test_no_calendar_arbitrage_single_node() -> None:
    assert make_ssvi().no_calendar_arbitrage()


def test_calendar_arbitrage_violated_by_rho_jump() -> None:
    # flat curvature leaves no room for a correlation change
    ssvi = SSVI(
        ttm=[Decimal("0.5"), Decimal("1.0")],
        theta=[Decimal("0.05"), Decimal("0.05")],
        psi=[Decimal("0.2"), Decimal("0.2")],
        rho=[Decimal("-0.5"), Decimal("0.5")],
    )
    assert not ssvi.no_calendar_arbitrage()


def test_no_calendar_arbitrage_implies_no_crossing() -> None:
    ssvi = make_essvi(["-0.4", "-0.35", "-0.2"])
    assert ssvi.no_calendar_arbitrage()
    for ttm1, ttm2 in [(0.25, 0.5), (0.5, 1.0), (0.3, 0.9)]:
        w1 = np.asarray(ssvi.total_variance(K, ttm1), dtype=float)
        w2 = np.asarray(ssvi.total_variance(K, ttm2), dtype=float)
        assert np.all(w2 >= w1)


# ---------------------------------------------------------------------------
# fit_surface
# ---------------------------------------------------------------------------


def _synthetic_surface(
    rhos: list[float],
    psis: list[float],
    thetas: list[float],
    ttms: list[float],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    slices = []
    for rho, psi, theta, ttm in zip(rhos, psis, thetas, ttms):
        ssvi = make_ssvi(rho=rho, psi=psi, theta=theta)
        slices.append((K, np.asarray(ssvi.iv(K, ttm), dtype=float), ttm))
    return slices


def test_fit_surface_recovers_parameters() -> None:
    rhos = [-0.3, -0.3, -0.3]
    psis = [0.12, 0.2, 0.26]
    thetas = [0.02, 0.05, 0.09]
    ttms = [0.25, 0.5, 1.0]
    fitted = SSVI.fit_surface(_synthetic_surface(rhos, psis, thetas, ttms))
    for node, expected in zip(fitted.rho, rhos):
        assert float(node) == pytest.approx(expected, abs=5e-3)
    for node, expected in zip(fitted.psi, psis):
        assert float(node) == pytest.approx(expected, rel=1e-2)
    for node, expected in zip(fitted.theta, thetas):
        assert float(node) == pytest.approx(expected, abs=1e-5)


def test_fit_surface_recovers_time_varying_correlation() -> None:
    rhos = [-0.5, -0.35, -0.2]
    fitted = SSVI.fit_surface(
        _synthetic_surface(
            rhos, [0.12, 0.2, 0.26], [0.02, 0.05, 0.09], [0.25, 0.5, 1.0]
        )
    )
    for node, expected in zip(fitted.rho, rhos):
        assert float(node) == pytest.approx(expected, abs=5e-3)


def test_fit_surface_reproduces_ivs() -> None:
    slices = _synthetic_surface(
        [-0.4, -0.35, -0.3], [0.12, 0.2, 0.26], [0.02, 0.05, 0.09], [0.25, 0.5, 1.0]
    )
    fitted = SSVI.fit_surface(slices)
    for k, iv_obs, ttm in slices:
        assert np.allclose(fitted.iv(k, ttm), iv_obs, atol=1e-3)


def test_fit_surface_no_static_arbitrage_by_construction() -> None:
    # decreasing input ATM variance is repaired by the calendar bounds
    slices = _synthetic_surface(
        [-0.3, -0.3, -0.3], [0.12, 0.2, 0.26], [0.02, 0.015, 0.09], [0.25, 0.5, 1.0]
    )
    fitted = SSVI.fit_surface(slices)
    thetas = [float(theta) for theta in fitted.theta]
    psis = [float(psi) for psi in fitted.psi]
    assert thetas == sorted(thetas)
    assert psis == sorted(psis)
    assert fitted.no_static_arbitrage()


def test_fit_surface_noisy_quotes_stay_arbitrage_free() -> None:
    rng = np.random.default_rng(7)
    slices = []
    for k, iv, ttm in _synthetic_surface(
        [-0.5, -0.4, -0.3], [0.12, 0.2, 0.26], [0.02, 0.05, 0.09], [0.25, 0.5, 1.0]
    ):
        slices.append((k, iv * (1 + rng.normal(0, 0.01, iv.shape)), ttm))
    fitted = SSVI.fit_surface(slices)
    assert fitted.no_static_arbitrage()


def test_fit_surface_requires_slices() -> None:
    with pytest.raises(ValueError, match="at least one maturity slice"):
        SSVI.fit_surface([])


def test_fit_surface_rejects_empty_slice() -> None:
    with pytest.raises(ValueError, match="at least one quote"):
        SSVI.fit_surface([(np.array([]), np.array([]), 0.5)])


def test_fit_surface_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        SSVI.fit_surface([(K, np.zeros(len(K) + 1), 0.5)])
