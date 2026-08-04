from __future__ import annotations

from decimal import Decimal

import pytest

from quantflow.options.parity import PutCallParities, PutCallParity
from quantflow.utils.price import Price


def _parity(strike: float, cp_mid: float, inverse: bool = False) -> PutCallParity:
    call = Price(bid=Decimal("1.0"), ask=Decimal("1.0"))
    put_value = Decimal(str(1.0 - cp_mid))
    put = Price(bid=put_value, ask=put_value)
    return PutCallParity(
        strike=Decimal(str(strike)), call=call, put=put, inverse=inverse
    )


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


def test_quote_discount_recovers_slope() -> None:
    da_true = 0.98
    dq_true = 0.95
    spot = 100
    strikes = [90, 100, 110, 120]
    mids = [spot * (da_true - dq_true * (k / spot)) for k in strikes]
    parities = PutCallParities.from_parities(
        [_parity(k, m) for k, m in zip(strikes, mids)], spot=spot, ttm=1
    )
    dq = parities.quote_discount(da_true / dq_true)
    assert dq == pytest.approx(dq_true, abs=1e-6)
