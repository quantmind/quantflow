from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from quantflow.rates.interest_rate import Rate
from quantflow.rates.nelson_siegel import NelsonSiegel

REF_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
ONE_YEAR = datetime(2025, 1, 1, tzinfo=timezone.utc)
TWO_YEARS = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------


def test_rate_from_number_stores_rate() -> None:
    r = Rate.from_number(0.05)
    assert float(r.rate) == pytest.approx(0.05, rel=1e-6)


def test_rate_percent() -> None:
    r = Rate.from_number(0.05)
    assert float(r.percent) == pytest.approx(5.0, rel=1e-6)


def test_rate_bps() -> None:
    r = Rate.from_number(0.0025)
    assert float(r.bps) == pytest.approx(25.0, rel=1e-6)


def test_rate_zero_default() -> None:
    r = Rate()
    assert r.rate == Decimal("0")
    assert float(r.percent) == 0.0
    assert float(r.bps) == 0.0


def test_discount_factor_continuous_one_year() -> None:
    rate = 0.05
    r = Rate.from_number(rate)
    df = r.discount_factor(REF_DATE, ONE_YEAR)
    expected = math.exp(-rate * 1.0)
    assert float(df) == pytest.approx(expected, rel=1e-6)


def test_discount_factor_continuous_two_years() -> None:
    rate = 0.03
    r = Rate.from_number(rate)
    df = r.discount_factor(REF_DATE, TWO_YEARS)
    expected = math.exp(-rate * 2.0)
    assert float(df) == pytest.approx(expected, rel=1e-5)


def test_discount_factor_zero_rate() -> None:
    r = Rate.from_number(0.0)
    df = r.discount_factor(REF_DATE, ONE_YEAR)
    assert float(df) == pytest.approx(1.0, rel=1e-9)


def test_discount_factor_expired_returns_one() -> None:
    r = Rate.from_number(0.05)
    # maturity before ref_date => ttm <= 0 => discount factor = 1
    df = r.discount_factor(ONE_YEAR, REF_DATE)
    assert df == Decimal("1")


def test_from_spot_and_forward_continuous() -> None:
    spot = Decimal("100")
    forward = Decimal("105")
    r = Rate.from_spot_and_forward(spot, forward, REF_DATE, ONE_YEAR)
    expected_rate = math.log(105 / 100) / 1.0
    assert float(r.rate) == pytest.approx(expected_rate, rel=1e-5)


def test_from_spot_and_forward_roundtrip() -> None:
    spot = Decimal("100")
    forward = Decimal("110")
    r = Rate.from_spot_and_forward(spot, forward, REF_DATE, TWO_YEARS)
    # applying the rate as a growth factor should recover the forward/spot ratio
    ttm = 2.0
    growth = math.exp(float(r.rate) * ttm)
    assert growth == pytest.approx(float(forward / spot), rel=1e-5)


def test_from_spot_and_forward_expired_returns_zero_rate() -> None:
    spot = Decimal("100")
    forward = Decimal("105")
    r = Rate.from_spot_and_forward(spot, forward, ONE_YEAR, REF_DATE)
    assert r.rate == Decimal("0")


# ---------------------------------------------------------------------------
# NelsonSiegel
# ---------------------------------------------------------------------------


def _flat_curve(level: float = 0.05) -> NelsonSiegel:
    """Flat curve: beta2=beta3=0, so yield = beta1 for all maturities."""
    return NelsonSiegel(
        beta1=Decimal(str(level)),
        beta2=Decimal("0"),
        beta3=Decimal("0"),
        lambda_=Decimal("1"),
    )


def test_nelson_siegel_flat_curve_discount_factor() -> None:
    ns = _flat_curve(0.05)
    ttm = 1.0
    df = ns.discount_factor(ttm)
    expected = math.exp(-0.05 * ttm)
    assert float(df) == pytest.approx(expected, rel=1e-5)


def test_nelson_siegel_flat_curve_two_year() -> None:
    ns = _flat_curve(0.04)
    df = ns.discount_factor(2.0)
    expected = math.exp(-0.04 * 2.0)
    assert float(df) == pytest.approx(expected, rel=1e-5)


def test_nelson_siegel_discount_factor_zero_ttm() -> None:
    ns = _flat_curve(0.05)
    assert ns.discount_factor(0) == Decimal("1")


def test_nelson_siegel_discount_factor_negative_ttm() -> None:
    ns = _flat_curve(0.05)
    assert ns.discount_factor(-1) == Decimal("1")


def test_nelson_siegel_instantaneous_forward_rate_at_zero() -> None:
    ns = NelsonSiegel(
        beta1=Decimal("0.04"),
        beta2=Decimal("0.02"),
        beta3=Decimal("0.01"),
        lambda_=Decimal("1"),
    )
    # at ttm=0: f(0) = beta1 + beta2
    fr = ns.instanteous_forward_rate(0)
    assert float(fr) == pytest.approx(0.06, rel=1e-6)


def test_nelson_siegel_instantaneous_forward_rate_large_ttm() -> None:
    ns = NelsonSiegel(
        beta1=Decimal("0.04"),
        beta2=Decimal("0.02"),
        beta3=Decimal("0.01"),
        lambda_=Decimal("1"),
    )
    # as ttm -> inf, e^{-ttm/lambda} -> 0, so f -> beta1
    fr = ns.instanteous_forward_rate(100)
    assert float(fr) == pytest.approx(0.04, abs=1e-5)


def test_nelson_siegel_discount_factor_increases_with_rate() -> None:
    # higher rate => smaller discount factor
    ns_low = _flat_curve(0.02)
    ns_high = _flat_curve(0.08)
    ttm = 1.0
    assert float(ns_low.discount_factor(ttm)) > float(ns_high.discount_factor(ttm))


def test_nelson_siegel_discount_factor_decreases_with_ttm() -> None:
    ns = _flat_curve(0.05)
    df1 = float(ns.discount_factor(1.0))
    df5 = float(ns.discount_factor(5.0))
    assert df1 > df5


def test_nelson_siegel_fit_recovers_parameters() -> None:
    ns_true = NelsonSiegel(
        beta1=Decimal("0.04"),
        beta2=Decimal("-0.02"),
        beta3=Decimal("0.03"),
        lambda_=Decimal("1.5"),
    )
    ttm = np.linspace(0.25, 10.0, 20)
    rates = np.array([float(ns_true.discount_factor(t)) for t in ttm])
    # convert discount factors back to zero rates for fitting
    zero_rates = -np.log(rates) / ttm
    ns_fit = NelsonSiegel.fit(ttm, zero_rates)
    for t in [1.0, 2.0, 5.0]:
        assert float(ns_fit.discount_factor(t)) == pytest.approx(
            float(ns_true.discount_factor(t)), rel=1e-4
        )


def test_nelson_siegel_fit_flat_curve() -> None:
    ttm = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    rates = np.full_like(ttm, 0.05)
    ns = NelsonSiegel.fit(ttm, rates)
    for t in ttm:
        assert float(ns.discount_factor(t)) == pytest.approx(
            math.exp(-0.05 * t), rel=1e-4
        )


def test_nelson_siegel_consistency_forward_and_discount() -> None:
    """Numerical check: f(tau) ≈ -d ln D / d tau."""
    ns = NelsonSiegel(
        beta1=Decimal("0.04"),
        beta2=Decimal("0.015"),
        beta3=Decimal("0.008"),
        lambda_=Decimal("2"),
    )
    ttm = 1.5
    h = 1e-5
    df_plus = float(ns.discount_factor(ttm + h))
    df_minus = float(ns.discount_factor(ttm - h))
    numerical_fwd = -(math.log(df_plus) - math.log(df_minus)) / (2 * h)
    analytic_fwd = float(ns.instanteous_forward_rate(ttm))
    assert numerical_fwd == pytest.approx(analytic_fwd, rel=1e-4)
