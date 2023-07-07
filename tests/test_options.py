import math

import numpy as np
import pytest

from quantflow.options import bs
from quantflow.options.surface import VolSurfaceLoader

VolLoader = VolSurfaceLoader[None]
a = np.asarray


@pytest.fixture
def vol_loader() -> VolLoader:
    return VolLoader()


def test_atm_black_pricing():
    price = bs.black_call(a(0), a(0.2), a(1))
    implied_vol = bs.implied_black_volatility(a(0), price, a(1), a(0.5))
    assert pytest.approx(implied_vol) == a(0.2)


def test_otm_black_pricing():
    price = bs.black_call(math.log(1.1), 0.25, 1)
    implied_vol = bs.implied_black_volatility(math.log(1.1), price, 1, 0.5)
    assert pytest.approx(implied_vol) == 0.25


def test_itm_black_pricing():
    price = bs.black_call(math.log(0.9), 0.25, 1)
    implied_vol = bs.implied_black_volatility(math.log(0.9), price, 1, 0.5)
    assert pytest.approx(implied_vol) == 0.25


def test_ditm_black_pricing():
    price = bs.black_call(math.log(0.6), 0.25, 1)
    assert pytest.approx(price, 0.01) == 0.4
    implied_vol = bs.implied_black_volatility(math.log(0.6), price, 1, 0.5)
    assert pytest.approx(implied_vol) == 0.25
