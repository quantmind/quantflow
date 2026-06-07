"""End-to-end test for the non-inverse (quote-currency) pricing path.

Builds a synthetic option chain from known Black-Scholes prices, feeds it
through [VolSurfaceLoader][quantflow.options.surface.VolSurfaceLoader] with
`inverse=False`, then checks that the loader recovers the forward via
put-call parity and that `bs()` inverts back to the input volatility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from quantflow.options.bs import black_price
from quantflow.options.inputs import DefaultVolSecurity, OptionType
from quantflow.options.surface import VolSurfaceLoader
from quantflow.rates.no_discount import NoDiscount

REF_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)
MATURITY = datetime(2026, 7, 2, tzinfo=timezone.utc)  # roughly 0.5y
FORWARD = 100.0
SIGMA = 0.25
STRIKES = (80.0, 90.0, 100.0, 110.0, 120.0)
HALF_SPREAD = Decimal("0.005")  # tiny tick around the mid, in quote currency


def _black_mid_usd(strike: float, call_put: int, ttm: float) -> Decimal:
    """Black price in quote currency: forward-space price times the forward."""
    log_strike = float(np.log(strike / FORWARD))
    pfs = float(black_price(np.asarray(log_strike), SIGMA, ttm, call_put).sum())
    return Decimal(str(pfs * FORWARD))


def _build_loader(ttm: float) -> VolSurfaceLoader:
    loader = VolSurfaceLoader(
        asset="TEST",
        quote_curve=NoDiscount(ref_date=REF_DATE),
        asset_curve=NoDiscount(ref_date=REF_DATE),
    )
    loader.add_spot(
        DefaultVolSecurity.spot(),
        bid=Decimal(str(FORWARD)),
        ask=Decimal(str(FORWARD)),
    )
    for strike in STRIKES:
        for option_type, call_put in (
            (OptionType.CALL, 1),
            (OptionType.PUT, -1),
        ):
            mid = _black_mid_usd(strike, call_put, ttm)
            loader.add_option(
                DefaultVolSecurity.option(),
                strike=Decimal(str(strike)),
                maturity=MATURITY,
                option_type=option_type,
                bid=mid - HALF_SPREAD,
                ask=mid + HALF_SPREAD,
                inverse=False,
            )
    return loader


def test_loader_recovers_forward_via_parity() -> None:
    """With matched call/put prices the implied forward equals the true forward."""
    loader = _build_loader(ttm=0.5)
    surface = loader.surface()
    cross = surface.maturities[0]
    assert float(cross.forward.mid) == pytest.approx(FORWARD, rel=1e-6)


def test_bs_recovers_input_volatility() -> None:
    """`bs()` inverts the synthetic non-inverse prices back to the input sigma."""
    loader = _build_loader(ttm=0.5)
    surface = loader.surface()
    ttm = surface.maturities[0].ttm(surface.ref_date)
    # rebuild prices at the actual ttm so the inversion is not biased by the
    # slight day-count drift from our nominal 0.5y target.
    loader = _build_loader(ttm=ttm)
    surface = loader.surface()
    surface.bs()
    options = list(surface.option_prices(converged=True))
    assert options, "expected converged options on the synthetic surface"
    for option in options:
        assert option.iv == pytest.approx(SIGMA, abs=5e-4)


def test_non_inverse_price_in_forward_space_matches_black() -> None:
    """`price_in_forward_space` is the Black forward-space price."""
    loader = _build_loader(ttm=0.5)
    surface = loader.surface()
    ttm = surface.maturities[0].ttm(surface.ref_date)
    loader = _build_loader(ttm=ttm)
    surface = loader.surface()
    for option in surface.option_prices():
        log_strike = float(option.log_strike)
        call_put = 1 if option.option_type.is_call() else -1
        expected = float(black_price(np.asarray(log_strike), SIGMA, ttm, call_put))
        assert float(option.price_in_forward_space) == pytest.approx(expected, abs=2e-4)
