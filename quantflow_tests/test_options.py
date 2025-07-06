import json
import math

import numpy as np
import pytest

from quantflow.options import bs
from quantflow.options.calibration import HestonCalibration
from quantflow.options.pricer import OptionPricer
from quantflow.options.surface import (
    OptionPrice,
    OptionType,
    VolSurface,
    VolSurfaceInputs,
    surface_from_inputs,
)
from quantflow.sp.heston import Heston
from quantflow_tests.utils import has_plotly

a = np.asarray
CROSS_SECTIONS = 8


@pytest.fixture
def heston() -> OptionPricer[Heston]:
    return OptionPricer(model=Heston.create(vol=0.5, kappa=1, sigma=0.8, rho=0))


@pytest.fixture
def vol_surface() -> VolSurface:
    with open("quantflow_tests/volsurface.json") as fp:
        return surface_from_inputs(VolSurfaceInputs(**json.load(fp)))


@pytest.mark.parametrize("ttm", [0.4, 0.8, 1.4, 2])
def test_atm_black_pricing(ttm):
    price = bs.black_call(0, 0.2, ttm)
    result = bs.implied_black_volatility(0, price, ttm, 0.5, 1)
    assert pytest.approx(result[0]) == 0.2


@pytest.mark.parametrize("ttm", [0.4, 0.8, 1.4, 2])
def test_otm_black_pricing(ttm):
    price = bs.black_call(math.log(1.1), 0.25, ttm)
    result = bs.implied_black_volatility(math.log(1.1), price, ttm, 0.5, 1)
    assert pytest.approx(result[0]) == 0.25


@pytest.mark.parametrize("ttm", [0.4, 0.8, 1.4, 2])
def test_itm_black_pricing(ttm):
    price = bs.black_call(math.log(0.9), 0.25, ttm)
    result = bs.implied_black_volatility(math.log(0.9), price, ttm, 0.5, 1)
    assert pytest.approx(result[0]) == 0.25


def test_ditm_black_pricing():
    price = bs.black_call(math.log(0.6), 0.25, 1)
    assert pytest.approx(price, 0.01) == 0.4
    result = bs.implied_black_volatility(math.log(0.6), price, 1, 0.5, 1)
    assert pytest.approx(result[0]) == 0.25


def test_vol_surface(vol_surface: VolSurface):
    assert vol_surface.ref_date
    ts = vol_surface.term_structure()
    assert len(ts) == CROSS_SECTIONS
    options = vol_surface.options_df()
    crosses = []
    for index in range(0, len(vol_surface.maturities)):
        crosses.append(vol_surface.options_df(index=index))
    assert len(crosses) == CROSS_SECTIONS
    assert len(options) == sum(len(cross) for cross in crosses)


def test_same_vol_surface(vol_surface: VolSurface):
    inputs = vol_surface.inputs()
    vol_surface2 = surface_from_inputs(inputs)
    assert vol_surface == vol_surface2


def test_black_vol(vol_surface: VolSurface):
    options = vol_surface.option_list(index=1)
    for option in options:
        assert option.price_time > 0

    options = vol_surface.bs(index=1)
    converged = [o for o in options if o.converged]
    assert converged
    # calculate the black price now
    prices = vol_surface.calc_bs_prices(index=1)
    assert len(converged) == len(prices)
    for o, price in zip(converged, prices):
        assert pytest.approx(float(o.price)) == price


def test_call_put_parity():
    option = OptionPrice.create(100).calculate_price()
    assert option.moneyness == 0
    assert option.price == option.call_price
    option2 = OptionPrice.create(100, option_type=OptionType.put).calculate_price()
    assert option2.price == option2.put_price
    assert option2.price == option.put_price
    assert option2.call_price == option.price


def test_call_put_parity_otm():
    option = OptionPrice.create(105, forward=100).calculate_price()
    assert option.moneyness > 0
    assert option.price == option.call_price
    option2 = OptionPrice.create(
        105, forward=100, option_type=OptionType.put
    ).calculate_price()
    assert option2.price == option2.put_price
    assert option2.price == pytest.approx(option.put_price)
    assert option2.call_price == pytest.approx(option.price)


def test_calibration_setup(vol_surface: VolSurface, heston: OptionPricer[Heston]):
    cal = HestonCalibration(pricer=heston, vol_surface=vol_surface)
    assert cal.ref_date == vol_surface.ref_date
    assert cal.options
    n = len(cal.options)
    vol_range = cal.implied_vol_range()
    assert vol_range.lb < vol_range.ub
    assert vol_range.lb > 0
    assert vol_range.ub < 10
    cal2 = cal.remove_implied_above(1.0)
    assert len(cal2.options) == n
    cal2 = cal.remove_implied_above(0.95)
    assert len(cal2.options) < n


def test_calibration(vol_surface: VolSurface, heston: OptionPricer[Heston]):
    vol_surface.maturities = vol_surface.maturities[1:]
    cal = HestonCalibration(
        pricer=heston, vol_surface=vol_surface
    ).remove_implied_above(0.95)
    cal.fit()
    if has_plotly:
        assert cal.plot(index=2) is not None
