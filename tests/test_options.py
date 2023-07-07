import json
import math

import numpy as np
import pytest

from quantflow.options import bs
from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs

a = np.asarray


@pytest.fixture
def vol_surface() -> VolSurface:
    with open("tests/volsurface.json") as fp:
        return surface_from_inputs(VolSurfaceInputs(**json.load(fp)))


@pytest.mark.parametrize("ttm", [0.4, 0.8, 1.4, 2])
def test_atm_black_pricing(ttm):
    price = bs.black_call(a(0), a(0.2), a(ttm))
    implied_vol = bs.implied_black_volatility(a(0), price, a(ttm), a(0.5))
    assert pytest.approx(implied_vol) == a(0.2)


@pytest.mark.parametrize("ttm", [0.4, 0.8, 1.4, 2])
def test_otm_black_pricing(ttm):
    price = bs.black_call(math.log(1.1), 0.25, ttm)
    implied_vol = bs.implied_black_volatility(math.log(1.1), price, ttm, 0.5)
    assert pytest.approx(implied_vol) == 0.25


@pytest.mark.parametrize("ttm", [0.4, 0.8, 1.4, 2])
def test_itm_black_pricing(ttm):
    price = bs.black_call(math.log(0.9), 0.25, ttm)
    implied_vol = bs.implied_black_volatility(math.log(0.9), price, ttm, 0.5)
    assert pytest.approx(implied_vol) == 0.25


def test_ditm_black_pricing():
    price = bs.black_call(math.log(0.6), 0.25, 1)
    assert pytest.approx(price, 0.01) == 0.4
    implied_vol = bs.implied_black_volatility(math.log(0.6), price, 1, 0.5)
    assert pytest.approx(implied_vol) == 0.25


def test_vol_surface(vol_surface: VolSurface):
    assert vol_surface.ref_date
    ts = vol_surface.term_structure()
    assert len(ts) == 8
    options = vol_surface.options_df()
    crosses = []
    for index in range(0, len(vol_surface.maturities)):
        crosses.append(vol_surface.options_df(index=index))
    assert len(crosses) == 8
    assert len(options) == sum(len(cross) for cross in crosses)


def test_same_vol_surface(vol_surface: VolSurface):
    inputs = vol_surface.inputs()
    vol_surface2 = surface_from_inputs(inputs)
    assert vol_surface == vol_surface2
