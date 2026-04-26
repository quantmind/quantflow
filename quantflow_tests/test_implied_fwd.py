"""Tests for ImpliedFwdPrice.aggregate"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantflow.options.inputs import DefaultVolSecurity
from quantflow.options.surface import FwdPrice, ImpliedFwdPrice

MATURITY = datetime(2026, 12, 31, tzinfo=timezone.utc)


def make_implied(
    mid: float, spread_bp: float, strike: float | None = None
) -> ImpliedFwdPrice:
    mid_d = Decimal(str(round(mid, 6)))
    half_spread = Decimal(str(round(mid * spread_bp / 20000, 8)))
    return ImpliedFwdPrice(
        security=DefaultVolSecurity.forward(),
        bid=mid_d - half_spread,
        ask=mid_d + half_spread,
        strike=Decimal(str(round(strike if strike is not None else mid, 6))),
        maturity=MATURITY,
    )


def make_fwd(mid: float, spread_bp: float) -> FwdPrice:
    mid_d = Decimal(str(round(mid, 6)))
    half_spread = Decimal(str(round(mid * spread_bp / 20000, 8)))
    return FwdPrice(
        security=DefaultVolSecurity.forward(),
        bid=mid_d - half_spread,
        ask=mid_d + half_spread,
        maturity=MATURITY,
    )


def test_aggregate_empty_no_default_returns_none() -> None:
    assert ImpliedFwdPrice.aggregate([], ttm=1.0) is None


def test_aggregate_empty_with_default_returns_default() -> None:
    default = make_fwd(100, 20)
    assert ImpliedFwdPrice.aggregate([], ttm=1.0, default=default) is default


def make_implied_market(
    theoretical_forward: float,
    mid_fraction: float,
    spread_multiplier: float,
    strike_fraction: float,
) -> ImpliedFwdPrice:
    """Implied forward whose spread scales with distance from theoretical_forward."""
    mid = theoretical_forward * mid_fraction
    spread_bp = max(
        1.0,
        abs(mid - theoretical_forward)
        / theoretical_forward
        * 10000
        * spread_multiplier,
    )
    return make_implied(mid, spread_bp, strike=theoretical_forward * strike_fraction)


def test_aggregate_all_invalid_returns_default() -> None:
    invalid = ImpliedFwdPrice(
        security=DefaultVolSecurity.forward(),
        bid=Decimal("101"),
        ask=Decimal("99"),  # bid > ask
        strike=Decimal("100"),
        maturity=MATURITY,
    )
    default = make_fwd(100, 20)
    assert ImpliedFwdPrice.aggregate([invalid], ttm=1.0, default=default) is default


@given(
    theoretical_forward=st.floats(
        min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False
    ),
    anchor=st.floats(
        min_value=0.9, max_value=0.99, allow_nan=False, allow_infinity=False
    ),
    mid_fractions=st.lists(
        st.floats(
            min_value=0.95, max_value=1.05, allow_nan=False, allow_infinity=False
        ),
        min_size=2,
        max_size=8,
    ),
    spread_multipliers=st.lists(
        st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    ),
    strike_fractions=st.lists(
        st.floats(min_value=0.8, max_value=1.2, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    ),
    ttm=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_aggregate_with_previous_forward(
    theoretical_forward: float,
    anchor: float,
    mid_fractions: list[float],
    spread_multipliers: list[float],
    strike_fractions: list[float],
    ttm: float,
) -> None:
    previous_forward = Decimal(str(round(theoretical_forward * anchor, 4)))
    n = min(len(mid_fractions), len(spread_multipliers), len(strike_fractions))
    forwards = [
        make_implied_market(
            theoretical_forward,
            mid_fractions[i],
            spread_multipliers[i],
            strike_fractions[i],
        )
        for i in range(n)
    ]
    result = ImpliedFwdPrice.aggregate(
        forwards, ttm=ttm, previous_forward=previous_forward
    )
    assert result is not None
    assert result.is_valid()
    mids = [float(f.mid) for f in forwards if f.is_valid()]
    assert min(mids) - 1e-4 <= float(result.mid) <= max(mids) + 1e-4


def test_aggregate_implied_tighter_than_default_uses_implied() -> None:
    # implied forwards (5 bp) are tighter than the default (20 bp)
    # the default is not included in the candidate pool, result comes from implied
    forwards = [make_implied(100, 5), make_implied(100, 5)]
    default = make_fwd(100, 20)
    result = ImpliedFwdPrice.aggregate(forwards, ttm=1.0, default=default)
    assert result is not None
    assert result is not default
    assert float(result.mid) == pytest.approx(100.0, rel=1e-3)


def test_aggregate_previous_forward_pulls_result_toward_anchor() -> None:
    # two forwards: one at 90 (the anchor), one at 110, same tight spread
    # without previous_forward: result ≈ 100 (equal weights)
    # with previous_forward=90: forward at 90 gets proximity_weight=1,
    #   forward at 110 is penalised → result pulled below 100
    fwd_at_anchor = make_implied(90, 5)
    fwd_above = make_implied(110, 5)
    previous_forward = Decimal("90")

    result_without = ImpliedFwdPrice.aggregate([fwd_at_anchor, fwd_above], ttm=1.0)
    result_with = ImpliedFwdPrice.aggregate(
        [fwd_at_anchor, fwd_above], ttm=1.0, previous_forward=previous_forward
    )
    assert result_without is not None
    assert result_with is not None
    assert float(result_with.mid) < float(result_without.mid)


def test_aggregate_outlier_with_wide_spread_does_not_move_result() -> None:
    # three tight forwards near 100, one outlier far away with enormous spread
    tight = [make_implied(100, 5) for _ in range(3)]
    outlier = make_implied(200, 500)
    result_without_outlier = ImpliedFwdPrice.aggregate(tight, ttm=1.0)
    result_with_outlier = ImpliedFwdPrice.aggregate(tight + [outlier], ttm=1.0)
    assert result_without_outlier is not None
    assert result_with_outlier is not None
    assert (
        abs(float(result_with_outlier.mid) - float(result_without_outlier.mid)) < 0.01
    )
