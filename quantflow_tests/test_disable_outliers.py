"""Tests for VolCrossSection.disable_outliers with synthetic cross sections."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np

from quantflow.options.inputs import DefaultVolSecurity
from quantflow.options.surface import (
    FwdPrice,
    OptionMetadata,
    OptionPrice,
    OptionPrices,
    OptionType,
    Side,
    Strike,
    VolCrossSection,
)
from quantflow.options.svi import SVI

MATURITY = datetime(2026, 12, 31, tzinfo=timezone.utc)
TTM = 1.0
FORWARD = Decimal("100")
SECURITY = DefaultVolSecurity.option()
FWD_SECURITY = DefaultVolSecurity.forward()


def _make_option(
    strike: float,
    iv_mid: float,
    iv_spread_fraction: float,
    option_type: OptionType = OptionType.call,
) -> OptionPrices[DefaultVolSecurity]:
    iv_bid = iv_mid * (1 - iv_spread_fraction / 2)
    iv_ask = iv_mid * (1 + iv_spread_fraction / 2)
    meta = OptionMetadata(
        strike=Decimal(str(strike)),
        option_type=option_type,
        maturity=MATURITY,
        forward=FORWARD,
        ttm=TTM,
        inverse=False,
    )
    bid = OptionPrice(
        price=Decimal("0.01"),
        meta=meta,
        implied_vol=iv_bid,
        side=Side.bid,
        converged=True,
    )
    ask = OptionPrice(
        price=Decimal("0.02"),
        meta=meta,
        implied_vol=iv_ask,
        side=Side.ask,
        converged=True,
    )
    return OptionPrices(security=SECURITY, meta=meta, bid=bid, ask=ask)


def _make_cross(
    strikes: list[float],
    iv_mids: list[float],
    iv_spread_fractions: list[float],
) -> VolCrossSection[DefaultVolSecurity]:
    strike_objs = [
        Strike(
            strike=Decimal(str(k)),
            call=_make_option(k, iv, sp),
        )
        for k, iv, sp in zip(strikes, iv_mids, iv_spread_fractions)
    ]
    fwd = FwdPrice(
        security=FWD_SECURITY,
        bid=FORWARD,
        ask=FORWARD,
        maturity=MATURITY,
    )
    return VolCrossSection(maturity=MATURITY, forward=fwd, strikes=tuple(strike_objs))


def _converged_count(cross: VolCrossSection) -> int:
    return sum(1 for _ in cross.option_securities(converged=True))


# ---------------------------------------------------------------------------
# pass 1: spread filter
# ---------------------------------------------------------------------------


def test_pass1_disables_wide_spread_option() -> None:
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    iv_mids = [0.25, 0.22, 0.20, 0.22, 0.25]
    # last option has spread fraction 0.4 — above the 0.2 threshold
    iv_spreads = [0.05, 0.05, 0.05, 0.05, 0.40]
    cross = _make_cross(strikes, iv_mids, iv_spreads)
    assert _converged_count(cross) == 5
    cross.disable_outliers(ttm=TTM, svi_residual_fraction=100.0)
    assert _converged_count(cross) == 4


def test_pass1_disables_all_options_above_threshold() -> None:
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    iv_mids = [0.25, 0.22, 0.20, 0.22, 0.25]
    iv_spreads = [0.50, 0.50, 0.05, 0.50, 0.50]
    cross = _make_cross(strikes, iv_mids, iv_spreads)
    cross.disable_outliers(ttm=TTM, svi_residual_fraction=100.0)
    assert _converged_count(cross) == 1


def test_pass1_keeps_options_below_threshold() -> None:
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    iv_mids = [0.25, 0.22, 0.20, 0.22, 0.25]
    # all at 0.10 fraction — well below the 0.20 threshold
    iv_spreads = [0.10, 0.10, 0.10, 0.10, 0.10]
    cross = _make_cross(strikes, iv_mids, iv_spreads)
    cross.disable_outliers(ttm=TTM, svi_residual_fraction=100.0)
    assert _converged_count(cross) == 5


def test_pass1_disables_zero_mid_vol() -> None:
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    # one option has zero mid vol
    iv_mids = [0.25, 0.22, 0.0, 0.22, 0.25]
    iv_spreads = [0.05, 0.05, 0.05, 0.05, 0.05]
    cross = _make_cross(strikes, iv_mids, iv_spreads)
    cross.disable_outliers(ttm=TTM, svi_residual_fraction=100.0)
    assert _converged_count(cross) == 4


# ---------------------------------------------------------------------------
# pass 2: SVI outlier filter
# ---------------------------------------------------------------------------


def _svi_smile(
    strikes: list[float],
    ttm: float = TTM,
    forward: float = float(FORWARD),
) -> list[float]:
    svi = SVI(
        a=Decimal("0.04"),
        b=Decimal("0.1"),
        rho=Decimal("-0.2"),
        m=Decimal("0.0"),
        theta=Decimal("0.15"),
    )
    k = np.log(np.array(strikes) / forward)
    return svi.implied_vol(k, ttm).tolist()


def test_pass2_disables_svi_outlier() -> None:
    strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    iv_mids = _svi_smile(strikes)
    # inject a large outlier at strike 100 (ATM)
    iv_mids[4] *= 2.0
    iv_spreads = [0.05] * len(strikes)
    cross = _make_cross(strikes, iv_mids, iv_spreads)
    assert _converged_count(cross) == 9
    cross.disable_outliers(
        ttm=TTM, bid_ask_spread_fraction=1.0, svi_residual_fraction=0.2
    )
    assert _converged_count(cross) == 8


def test_pass2_clean_smile_not_disabled() -> None:
    strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    iv_mids = _svi_smile(strikes)
    iv_spreads = [0.05] * len(strikes)
    cross = _make_cross(strikes, iv_mids, iv_spreads)
    cross.disable_outliers(
        ttm=TTM, bid_ask_spread_fraction=1.0, svi_residual_fraction=0.2
    )
    assert _converged_count(cross) == 9


def test_pass2_does_not_run_with_fewer_than_5_options() -> None:
    strikes = [95.0, 97.5, 100.0, 102.5, 105.0]
    iv_mids = _svi_smile(strikes)
    # discard one so only 4 survive pass 1
    iv_spreads = [0.05, 0.05, 0.05, 0.05, 0.90]
    cross = _make_cross(strikes, iv_mids, iv_spreads)
    # 4 remain after pass 1 — pass 2 should not run
    cross.disable_outliers(ttm=TTM, svi_residual_fraction=0.01)
    # the remaining 4 should be untouched by pass 2
    assert _converged_count(cross) == 4


def test_pass2_repeat_removes_multiple_outliers() -> None:
    strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    iv_mids = _svi_smile(strikes)
    # inject two large outliers
    iv_mids[1] *= 3.0
    iv_mids[7] *= 3.0
    iv_spreads = [0.05] * len(strikes)
    cross = _make_cross(strikes, iv_mids, iv_spreads)
    cross.disable_outliers(
        ttm=TTM, bid_ask_spread_fraction=1.0, svi_residual_fraction=0.2, repeat=3
    )
    # at least the two injected outliers must be removed
    assert _converged_count(cross) <= 7


def test_pass2_huge_threshold_removes_nothing() -> None:
    strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    iv_mids = _svi_smile(strikes)
    iv_mids[4] *= 5.0  # extreme outlier
    iv_spreads = [0.05] * len(strikes)
    cross = _make_cross(strikes, iv_mids, iv_spreads)
    cross.disable_outliers(
        ttm=TTM, bid_ask_spread_fraction=1.0, svi_residual_fraction=100.0
    )
    assert _converged_count(cross) == 9
