from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest

from quantflow.rates.nelson_siegel import NelsonSiegel


def _flat_curve(level: float = 0.05) -> NelsonSiegel:
    return NelsonSiegel(
        beta1=Decimal(str(level)),
        beta2=Decimal("0"),
        beta3=Decimal("0"),
        lambda_=Decimal("1"),
    )


def _true_curve() -> NelsonSiegel:
    return NelsonSiegel(
        beta1=Decimal("0.04"),
        beta2=Decimal("-0.02"),
        beta3=Decimal("0.03"),
        lambda_=Decimal("1.5"),
    )


# ---------------------------------------------------------------------------
# Discount factor and forward rate
# ---------------------------------------------------------------------------


def test_flat_curve_discount_factor_one_year() -> None:
    ns = _flat_curve(0.05)
    assert float(ns.discount_factor(1.0)) == pytest.approx(math.exp(-0.05), rel=1e-5)


def test_flat_curve_discount_factor_two_year() -> None:
    ns = _flat_curve(0.04)
    assert float(ns.discount_factor(2.0)) == pytest.approx(math.exp(-0.08), rel=1e-5)


def test_discount_factor_zero_ttm() -> None:
    assert _flat_curve(0.05).discount_factor(0) == Decimal("1")


def test_discount_factor_negative_ttm() -> None:
    assert _flat_curve(0.05).discount_factor(-1) == Decimal("1")


def test_discount_factor_increases_with_lower_rate() -> None:
    ttm = 1.0
    assert float(_flat_curve(0.02).discount_factor(ttm)) > float(
        _flat_curve(0.08).discount_factor(ttm)
    )


def test_discount_factor_decreases_with_ttm() -> None:
    ns = _flat_curve(0.05)
    assert float(ns.discount_factor(1.0)) > float(ns.discount_factor(5.0))


def test_instantaneous_forward_rate_at_zero() -> None:
    ns = NelsonSiegel(
        beta1=Decimal("0.04"),
        beta2=Decimal("0.02"),
        beta3=Decimal("0.01"),
        lambda_=Decimal("1"),
    )
    assert float(ns.instantaneous_forward_rate(0)) == pytest.approx(0.06, rel=1e-6)


def test_instantaneous_forward_rate_large_ttm() -> None:
    ns = NelsonSiegel(
        beta1=Decimal("0.04"),
        beta2=Decimal("0.02"),
        beta3=Decimal("0.01"),
        lambda_=Decimal("1"),
    )
    assert float(ns.instantaneous_forward_rate(100)) == pytest.approx(0.04, abs=1e-5)


def test_consistency_forward_and_discount() -> None:
    ns = NelsonSiegel(
        beta1=Decimal("0.04"),
        beta2=Decimal("0.015"),
        beta3=Decimal("0.008"),
        lambda_=Decimal("2"),
    )
    ttm, h = 1.5, 1e-5
    numerical = -(
        math.log(float(ns.discount_factor(ttm + h)))
        - math.log(float(ns.discount_factor(ttm - h)))
    ) / (2 * h)
    assert numerical == pytest.approx(
        float(ns.instantaneous_forward_rate(ttm)), rel=1e-4
    )


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


def test_jacobian_shape() -> None:
    ns = _true_curve()
    ttm = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    J = ns.jacobian(ttm)
    assert J.shape == (len(ttm), 4)


def test_jacobian_matches_finite_differences() -> None:
    ns = _true_curve()
    ttm = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    h = 1e-5
    params = [ns.beta1, ns.beta2, ns.beta3, ns.lambda_]
    fields = ["beta1", "beta2", "beta3", "lambda_"]
    J_analytical = ns.jacobian(ttm)
    J_numerical = np.zeros_like(J_analytical)
    for i, (field, p) in enumerate(zip(fields, params)):
        ns_fwd = ns.model_copy(update={field: Decimal(str(float(p) + h))})
        ns_bwd = ns.model_copy(update={field: Decimal(str(float(p) - h))})
        df_fwd = np.array([float(ns_fwd.discount_factor(t)) for t in ttm])
        df_bwd = np.array([float(ns_bwd.discount_factor(t)) for t in ttm])
        J_numerical[:, i] = (df_fwd - df_bwd) / (2 * h)
    np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-4)


# ---------------------------------------------------------------------------
# calibrate — clean data
# ---------------------------------------------------------------------------


def test_calibrate_recovers_curve_noiseless() -> None:
    ns_true = _true_curve()
    ttm = np.linspace(0.25, 10.0, 20)
    rates = -np.log([float(ns_true.discount_factor(t)) for t in ttm]) / ttm
    ns_fit = NelsonSiegel().calibrator().calibrate(ttm, rates)
    for t in [1.0, 2.0, 5.0]:
        assert float(ns_fit.discount_factor(t)) == pytest.approx(
            float(ns_true.discount_factor(t)), rel=1e-4
        )


def test_calibrate_flat_curve() -> None:
    ttm = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    rates = np.full_like(ttm, 0.05)
    ns = NelsonSiegel().calibrator().calibrate(ttm, rates)
    for t in ttm:
        assert float(ns.discount_factor(t)) == pytest.approx(
            math.exp(-0.05 * t), rel=1e-4
        )


# ---------------------------------------------------------------------------
# calibrate — robustness
# ---------------------------------------------------------------------------

# Crypto-realistic TTM grid: short maturities out to ~2 years
_CRYPTO_TTMS = np.array([1 / 52, 2 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0, 2.0])


def _true_rates(ns: NelsonSiegel, ttm: np.ndarray) -> np.ndarray:
    return -np.log([float(ns.discount_factor(t)) for t in ttm]) / ttm


def _df_rmse(ns_fit: NelsonSiegel, ns_true: NelsonSiegel, ttm: np.ndarray) -> float:
    fitted = np.array([float(ns_fit.discount_factor(t)) for t in ttm])
    true = np.array([float(ns_true.discount_factor(t)) for t in ttm])
    return float(np.sqrt(np.mean((fitted - true) ** 2)))


def test_calibrate_crypto_ttms_noiseless() -> None:
    ns_true = _true_curve()
    rates = _true_rates(ns_true, _CRYPTO_TTMS)
    ns_fit = NelsonSiegel().calibrator().calibrate(_CRYPTO_TTMS, rates)
    assert _df_rmse(ns_fit, ns_true, _CRYPTO_TTMS) < 1e-4


def test_calibrate_with_gaussian_noise() -> None:
    rng = np.random.default_rng(42)
    ns_true = _true_curve()
    rates = _true_rates(ns_true, _CRYPTO_TTMS)
    noisy = rates + rng.normal(0, 0.002, size=len(rates))
    ns_fit = NelsonSiegel().calibrator().calibrate(_CRYPTO_TTMS, noisy)
    assert _df_rmse(ns_fit, ns_true, _CRYPTO_TTMS) < 0.005
