from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from quantflow.options.svi import SVI

TTM = 1.0
K = np.linspace(-0.5, 0.5, 21)


def make_svi(**kwargs: float) -> SVI:
    defaults = dict(a=0.04, b=0.1, rho=-0.3, m=0.0, theta=0.2)
    defaults.update(kwargs)
    return SVI(**{k: Decimal(str(v)) for k, v in defaults.items()})


# ---------------------------------------------------------------------------
# total_variance
# ---------------------------------------------------------------------------


def test_total_variance_is_positive() -> None:
    svi = make_svi()
    w = svi.total_variance(K)
    assert np.all(w > 0)


def test_total_variance_symmetric_when_rho_zero_m_zero() -> None:
    svi = make_svi(rho=0.0, m=0.0)
    w = svi.total_variance(K)
    assert np.allclose(w, w[::-1])


def test_total_variance_minimum_at_m() -> None:
    m = 0.1
    svi = make_svi(m=m)
    k_dense = np.linspace(-1.0, 1.0, 1000)
    w = svi.total_variance(k_dense)
    assert pytest.approx(k_dense[np.argmin(w)], abs=0.01) == m + float(svi.rho) * float(
        svi.theta
    ) / np.sqrt(1 - float(svi.rho) ** 2) * (-1)


def test_total_variance_scalar_input() -> None:
    svi = make_svi()
    w = svi.total_variance(0.0)
    assert w.shape == (1,) or w.ndim == 0


# ---------------------------------------------------------------------------
# implied_vol
# ---------------------------------------------------------------------------


def test_implied_vol_non_negative() -> None:
    svi = make_svi()
    iv = svi.implied_vol(K, TTM)
    assert np.all(iv >= 0)


def test_implied_vol_zero_where_variance_non_positive() -> None:
    # force a very negative a so some w(k) <= 0
    svi = make_svi(a=-0.5, b=0.01)
    iv = svi.implied_vol(K, TTM)
    w = svi.total_variance(K)
    assert np.all(iv[w <= 0] == 0.0)


def test_implied_vol_consistent_with_total_variance() -> None:
    svi = make_svi()
    iv = svi.implied_vol(K, TTM)
    w = svi.total_variance(K)
    assert np.allclose(iv**2 * TTM, w)


def test_implied_vol_scales_with_ttm() -> None:
    svi = make_svi()
    iv1 = svi.implied_vol(K, 1.0)
    iv2 = svi.implied_vol(K, 0.25)
    # same total variance, shorter ttm => higher iv
    assert np.all(iv2 > iv1)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def _synthetic_iv(svi: SVI, k: np.ndarray, ttm: float) -> np.ndarray:
    return svi.implied_vol(k, ttm)


def test_fit_recovers_parameters() -> None:
    true_svi = make_svi(a=0.04, b=0.1, rho=-0.2, m=0.05, theta=0.15)
    iv_obs = _synthetic_iv(true_svi, K, TTM)
    fitted = SVI.fit(K, iv_obs, TTM)
    assert float(fitted.a) == pytest.approx(float(true_svi.a), abs=1e-4)
    assert float(fitted.b) == pytest.approx(float(true_svi.b), abs=1e-4)
    assert float(fitted.rho) == pytest.approx(float(true_svi.rho), abs=1e-4)
    assert float(fitted.m) == pytest.approx(float(true_svi.m), abs=1e-4)
    assert float(fitted.theta) == pytest.approx(float(true_svi.theta), abs=1e-4)


def test_fit_reproduces_implied_vols() -> None:
    true_svi = make_svi()
    iv_obs = _synthetic_iv(true_svi, K, TTM)
    fitted = SVI.fit(K, iv_obs, TTM)
    iv_fit = fitted.implied_vol(K, TTM)
    assert np.allclose(iv_fit, iv_obs, atol=1e-5)


def test_fit_flat_smile() -> None:
    iv_obs = np.full_like(K, 0.20)
    fitted = SVI.fit(K, iv_obs, TTM)
    iv_fit = fitted.implied_vol(K, TTM)
    assert np.allclose(iv_fit, 0.20, atol=1e-4)


def test_fit_respects_bounds() -> None:
    true_svi = make_svi()
    iv_obs = _synthetic_iv(true_svi, K, TTM)
    fitted = SVI.fit(K, iv_obs, TTM)
    assert float(fitted.b) >= 0
    assert -1 < float(fitted.rho) < 1
    assert float(fitted.theta) > 0


def test_fit_skewed_smile() -> None:
    true_svi = make_svi(rho=-0.5, m=-0.1)
    iv_obs = _synthetic_iv(true_svi, K, TTM)
    fitted = SVI.fit(K, iv_obs, TTM)
    iv_fit = fitted.implied_vol(K, TTM)
    assert np.allclose(iv_fit, iv_obs, atol=1e-5)
