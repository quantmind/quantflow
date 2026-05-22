from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quantflow.rates.interest_rate import Rate

REF_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
ONE_YEAR = datetime(2025, 1, 1, tzinfo=timezone.utc)
TWO_YEARS = datetime(2026, 1, 1, tzinfo=timezone.utc)


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
