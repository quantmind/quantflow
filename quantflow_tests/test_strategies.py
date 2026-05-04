from datetime import datetime, timezone

import pytest

from quantflow.options.pricer import OptionPricer
from quantflow.options.strategies import (
    Butterfly,
    CalendarSpread,
    Spread,
    Straddle,
    Strangle,
)
from quantflow.sp.wiener import WienerProcess

REF_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
MATURITY = datetime(2025, 1, 1, tzinfo=timezone.utc)
FAR_MATURITY = datetime(2026, 1, 1, tzinfo=timezone.utc)
FORWARD = 100.0


@pytest.fixture
def pricer() -> OptionPricer:
    return OptionPricer(model=WienerProcess(sigma=0.3))


def test_straddle_from_strike(pricer: OptionPricer) -> None:
    p = Straddle.create(105.0, MATURITY).price(pricer, FORWARD, REF_DATE)
    assert p.price > 0
    assert p.gamma > 0


def test_strangle_from_moneyness(pricer: OptionPricer) -> None:
    p = Strangle.from_moneyness(FORWARD, MATURITY).price(pricer, FORWARD, REF_DATE)
    assert p.price > 0
    assert p.gamma > 0
    assert (
        p.price
        < Straddle.create(FORWARD, MATURITY).price(pricer, FORWARD, REF_DATE).price
    )
    assert Strangle.description != ""


def test_strangle_from_strikes(pricer: OptionPricer) -> None:
    p = Strangle.from_strikes(95.0, 105.0, MATURITY).price(pricer, FORWARD, REF_DATE)
    assert p.price > 0
    assert p.gamma > 0


def test_butterfly_from_moneyness(pricer: OptionPricer) -> None:
    p = Butterfly.from_moneyness(FORWARD, MATURITY).price(pricer, FORWARD, REF_DATE)
    assert p.price > 0
    assert p.gamma < 0
    assert Butterfly.description != ""


def test_butterfly_from_strikes(pricer: OptionPricer) -> None:
    p = Butterfly.from_strikes(95.0, 100.0, 105.0, MATURITY, FORWARD).price(
        pricer, FORWARD, REF_DATE
    )
    assert p.price > 0
    assert p.gamma < 0


def test_call_spread(pricer: OptionPricer) -> None:
    p = Spread.call(95.0, 105.0, MATURITY).price(pricer, FORWARD, REF_DATE)
    assert p.price > 0
    assert p.delta > 0


def test_put_spread(pricer: OptionPricer) -> None:
    p = Spread.put(95.0, 105.0, MATURITY).price(pricer, FORWARD, REF_DATE)
    assert p.price > 0
    assert p.delta < 0


def test_calendar_spread_call_below_forward(pricer: OptionPricer) -> None:
    # K < F: net delta negative when long (both calls and puts)
    p = CalendarSpread.call(80.0, MATURITY, FAR_MATURITY).price(
        pricer, FORWARD, REF_DATE
    )
    assert p.price > 0
    assert p.delta < 0


def test_calendar_spread_call_above_forward(pricer: OptionPricer) -> None:
    # K > F: net delta positive when long (both calls and puts)
    p = CalendarSpread.call(120.0, MATURITY, FAR_MATURITY).price(
        pricer, FORWARD, REF_DATE
    )
    assert p.price > 0
    assert p.delta > 0


def test_calendar_spread_put_below_forward(pricer: OptionPricer) -> None:
    # K < F: net delta negative when long (same as call)
    p = CalendarSpread.put(80.0, MATURITY, FAR_MATURITY).price(
        pricer, FORWARD, REF_DATE
    )
    assert p.price > 0
    assert p.delta < 0


def test_calendar_spread_put_above_forward(pricer: OptionPricer) -> None:
    # K > F: net delta positive when long (same as call)
    p = CalendarSpread.put(120.0, MATURITY, FAR_MATURITY).price(
        pricer, FORWARD, REF_DATE
    )
    assert p.price > 0
    assert p.delta > 0
