from decimal import Decimal

import pytest

from quantflow.options.pricer import OptionPricer, OptionType
from quantflow.sp.heston import HestonJ
from quantflow.sp.weiner import WeinerProcess
from quantflow.utils.distributions import DoubleExponential
from quantflow_tests.utils import has_plotly


@pytest.fixture
def pricer() -> OptionPricer[HestonJ[DoubleExponential]]:
    return OptionPricer(
        model=HestonJ.create(DoubleExponential, vol=0.5, kappa=1, sigma=0.8, rho=0)
    )


@pytest.mark.skipif(not has_plotly, reason="Plotly not installed")
def test_plot_surface(pricer: OptionPricer):
    fig = pricer.plot3d()
    surface = fig.data[0]
    assert surface.x is not None
    assert surface.y is not None
    assert surface.z is not None


def test_price_call(pricer: OptionPricer):
    price = pricer.price(
        option_type=OptionType.call,
        strike=100,
        forward=100,
        ttm=1.0,
    )
    assert price.strike == Decimal(100)
    assert price.forward == Decimal(100)
    assert price.moneyness == 0.0
    black = price.black
    assert black.iv < 0.5
    assert black.price == pytest.approx(price.price)


@pytest.mark.parametrize("strike,forward", [(90, 100), (100, 100), (110, 100)])
def test_weiner_matches_black(strike: int, forward: int) -> None:
    sigma = 0.3
    pricer = OptionPricer(model=WeinerProcess(sigma=sigma))
    price = pricer.price(
        option_type=OptionType.call, strike=strike, forward=forward, ttm=1.0
    )
    black = price.black
    assert float(black.iv) == pytest.approx(sigma, rel=1e-3)
    assert price.delta == pytest.approx(float(black.delta), rel=1e-3)
    assert price.gamma == pytest.approx(float(black.gamma), rel=5e-3)
