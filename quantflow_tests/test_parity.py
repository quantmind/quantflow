from __future__ import annotations

from decimal import Decimal

import pytest

from quantflow.options.parity import PutCallParity, PutCallParities
from quantflow.utils.price import Price


def _parity(strike: float, cp_mid: float, inverse: bool = False) -> PutCallParity:
    call = Price(bid=Decimal("1.0"), ask=Decimal("1.0"))
    put_value = Decimal(str(1.0 - cp_mid))
    put = Price(bid=put_value, ask=put_value)
    return PutCallParity(strike=Decimal(str(strike)), call=call, put=put, inverse=inverse)


def test_regressand_and_regressor_direct() -> None:
    parities = PutCallParities.from_parities(
        [_parity(90, 0.2), _parity(110, 0.0)], spot=100, ttm=1
    )
    y = parities.regressand()
    x = parities.regressor()
    assert y[0] == pytest.approx(0.002)
    assert y[1] == pytest.approx(0.0)
    assert x[0] == pytest.approx(0.9)
    assert x[1] == pytest.approx(1.1)


def test_regressand_inverse() -> None:
    parities = PutCallParities.from_parities(
        [_parity(90, 0.2, inverse=True), _parity(110, 0.0, inverse=True)],
        spot=100,
        ttm=1,
    )
    y = parities.regressand()
    assert y[0] == pytest.approx(0.2)
    assert y[1] == pytest.approx(0.0)


def test_fit_discounts_with_fixed_values() -> None:
    parities = PutCallParities.from_parities(
        [_parity(90, 12.5), _parity(110, -6.5)],
        100,
        1,
    )
    fitted_both = parities.fit_discounts(dq=0.95, da=0.98)
    assert fitted_both is not None
    assert fitted_both.quote_discount == pytest.approx(0.95)
    assert fitted_both.asset_discount == pytest.approx(0.98)

    fitted_da = parities.fit_discounts(dq=0.95)
    assert fitted_da is not None
    assert fitted_da.asset_discount == pytest.approx(0.98)

    fitted_dq = parities.fit_discounts(da=0.98)
    assert fitted_dq is not None
    assert fitted_dq.quote_discount == pytest.approx(0.95)


def test_fit_discounts_constrained_branch() -> None:
    da_true = 0.98
    dq_true = 0.95
    spot = 100
    strikes = [90, 100, 110, 120]
    mids = [spot * (da_true - dq_true * (k / spot)) for k in strikes]
    parities = PutCallParities.from_parities(
        [_parity(k, m) for k, m in zip(strikes, mids)], spot=spot, ttm=1
    )
    fitted = parities.fit_discounts()
    assert fitted is not None
    assert fitted.asset_discount == pytest.approx(da_true, abs=1e-6)
    assert fitted.quote_discount == pytest.approx(dq_true, abs=1e-6)


def test_fit_discounts_invalid_or_empty_returns_none() -> None:
    empty = PutCallParities.from_parities([], spot=100, ttm=1)
    assert empty.fit_discounts() is None

    parities = PutCallParities.from_parities([_parity(100, 2.0)], spot=100, ttm=1)
    assert parities.fit_discounts(dq=1.0, min_rate_q=0.1, min_rate_a=0.1) is None
