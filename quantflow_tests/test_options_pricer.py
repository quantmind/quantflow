from decimal import Decimal

import pytest

from quantflow.dists.distributions1d import DoubleExponential
from quantflow.options.pricer import OptionPricer, OptionType
from quantflow.sp.heston import Heston, HestonJ
from quantflow.sp.wiener import WienerProcess
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
        option_type=OptionType.CALL,
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
def test_wiener_matches_black(strike: int, forward: int) -> None:
    sigma = 0.3
    pricer = OptionPricer(model=WienerProcess(sigma=sigma))
    price = pricer.price(
        option_type=OptionType.CALL, strike=strike, forward=forward, ttm=1.0
    )
    black = price.black
    assert float(black.iv) == pytest.approx(sigma, rel=1e-3)
    assert price.delta == pytest.approx(float(black.delta), rel=1e-3)
    assert price.gamma == pytest.approx(float(black.gamma), rel=5e-3)


@pytest.mark.parametrize("strike,forward", [(80, 100), (100, 100), (120, 100)])
def test_put_call_parity_across_strikes(strike: int, forward: int) -> None:
    """`c - p = 1 - K/F` in forward space, on both sides of the forward.

    Regression for `as_option_type` previously using the clipped intrinsic,
    which collapsed the put price onto the call whenever the call was OTM.
    """
    pricer = OptionPricer(model=Heston.create(vol=0.2, kappa=2.0, sigma=0.5, rho=-0.5))
    call = pricer.price(
        option_type=OptionType.CALL, strike=strike, forward=forward, ttm=0.5
    )
    put = pricer.price(
        option_type=OptionType.PUT, strike=strike, forward=forward, ttm=0.5
    )
    assert call.price - put.price == pytest.approx(1.0 - strike / forward, abs=1e-9)
    assert call.delta - put.delta == pytest.approx(1.0, abs=1e-9)
    assert call.gamma == pytest.approx(put.gamma, abs=1e-9)


@pytest.mark.parametrize(
    "option_type,strike,forward,expected",
    [
        # calls: payoff max(F - K, 0) / F = max(0, 1 - K/F)
        (OptionType.CALL, 80, 100, 0.2),  # ITM
        (OptionType.CALL, 100, 100, 0.0),  # ATM
        (OptionType.CALL, 120, 100, 0.0),  # OTM
        # puts: payoff max(K - F, 0) / F = max(0, K/F - 1)
        (OptionType.PUT, 80, 100, 0.0),  # OTM
        (OptionType.PUT, 100, 100, 0.0),  # ATM
        (OptionType.PUT, 120, 100, 0.2),  # ITM
    ],
)
def test_intrinsic_value(
    option_type: OptionType, strike: int, forward: int, expected: float
) -> None:
    """`intrinsic_value` is the forward-space payoff if exercised immediately."""
    pricer = OptionPricer(model=Heston.create(vol=0.2, kappa=2.0, sigma=0.5, rho=-0.5))
    price = pricer.price(
        option_type=option_type, strike=strike, forward=forward, ttm=0.5
    )
    assert price.intrinsic_value == pytest.approx(expected, abs=1e-12)


def test_price_in_quote_scales_with_forward() -> None:
    """`price_in_quote` is the forward-space price multiplied by the forward."""
    pricer = OptionPricer(model=Heston.create(vol=0.2, kappa=2.0, sigma=0.5, rho=-0.5))
    price = pricer.price(
        option_type=OptionType.CALL, strike=5500, forward=5000, ttm=0.5
    )
    assert price.price_in_quote == pytest.approx(price.price * 5000.0, abs=1e-9)


@pytest.mark.parametrize("strike,forward", [(80, 100), (100, 100), (120, 100)])
def test_as_option_type_roundtrip(strike: int, forward: int) -> None:
    """`call.as_option_type(put).as_option_type(call)` recovers the original."""
    pricer = OptionPricer(model=Heston.create(vol=0.2, kappa=2.0, sigma=0.5, rho=-0.5))
    call = pricer.price(
        option_type=OptionType.CALL, strike=strike, forward=forward, ttm=0.5
    )
    roundtrip = call.as_option_type(OptionType.PUT).as_option_type(OptionType.CALL)
    assert roundtrip.price == pytest.approx(call.price, abs=1e-12)
    assert roundtrip.delta == pytest.approx(call.delta, abs=1e-12)
    assert roundtrip.gamma == pytest.approx(call.gamma, abs=1e-12)
