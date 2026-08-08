import numpy as np
import pytest

from quantflow.options.inputs import OptionType
from quantflow.options.pricer import OptionPricer, OptionPricingMethod
from quantflow.sp.heston import Heston
from quantflow.sp.rough_heston import RoughHeston
from quantflow_tests.utils import characteristic_tests


@pytest.fixture
def rough_heston() -> RoughHeston:
    return RoughHeston.create(
        vol=0.5, kappa=1.5, sigma=0.7, rho=-0.5, hurst=0.1, adams_steps=200
    )


def test_alpha(rough_heston: RoughHeston) -> None:
    assert rough_heston.alpha == pytest.approx(0.6)
    assert rough_heston.hurst == 0.1


def test_characteristic(rough_heston: RoughHeston) -> None:
    assert rough_heston.variance_process.is_positive is True
    assert rough_heston.characteristic(1, 0) == 1
    m = rough_heston.marginal(0.5)
    characteristic_tests(m)
    assert m.mean() == pytest.approx(0.0, abs=1e-6)


def test_scalar_time_required(rough_heston: RoughHeston) -> None:
    with pytest.raises(ValueError):
        rough_heston.characteristic_exponent(np.array([0.5, 1.0]), 1.0)


def test_heston_limit_characteristic() -> None:
    """At H=0.5 the rough variance is Markovian and the model is Heston."""
    kwargs = dict(vol=0.5, kappa=2.0, sigma=0.6, rho=-0.4)
    rough = RoughHeston.create(hurst=0.5, adams_steps=300, **kwargs)
    heston = Heston.create(**kwargs)
    mr = rough.marginal(1.0)
    mh = heston.marginal(1.0)
    u = np.linspace(0.0, 8.0, 40)
    # the martingale-corrected characteristic functions must agree
    np.testing.assert_allclose(
        mr.characteristic_corrected(u), mh.characteristic_corrected(u), atol=1e-3
    )
    assert mr.std() == pytest.approx(mh.std(), rel=1e-3)


def test_heston_limit_prices() -> None:
    """At H=0.5 the COS option prices must match the classical Heston model."""
    kwargs = dict(vol=0.5, kappa=2.0, sigma=0.6, rho=-0.4)
    rough = OptionPricer(
        model=RoughHeston.create(hurst=0.5, adams_steps=300, **kwargs),
        method=OptionPricingMethod.COS,
    )
    heston = OptionPricer(model=Heston.create(**kwargs), method=OptionPricingMethod.COS)
    for strike in (70.0, 85.0, 100.0, 115.0, 130.0):
        pr = rough.price(
            option_type=OptionType.CALL, strike=strike, forward=100.0, ttm=1.0
        )
        ph = heston.price(
            option_type=OptionType.CALL, strike=strike, forward=100.0, ttm=1.0
        )
        assert pr.price == pytest.approx(ph.price, abs=1e-4)


def test_adams_convergence() -> None:
    """Prices are stable once the fractional Adams grid is fine enough."""
    kwargs = dict(vol=0.5, kappa=1.5, sigma=0.7, rho=-0.5, hurst=0.2)
    prices = []
    for steps in (100, 200, 400):
        pricer = OptionPricer(
            model=RoughHeston.create(adams_steps=steps, **kwargs),
            method=OptionPricingMethod.COS,
        )
        prices.append(
            pricer.price(
                option_type=OptionType.CALL, strike=100.0, forward=100.0, ttm=0.5
            ).price
        )
    assert prices[2] == pytest.approx(prices[1], abs=1e-4)
    assert prices[1] == pytest.approx(prices[0], abs=1e-3)


def test_rough_skew_steeper_than_heston() -> None:
    """A rough model (H<1/2) produces a steeper short-maturity skew than Heston."""
    kwargs = dict(vol=0.5, kappa=1.5, sigma=0.7, rho=-0.6)
    rough = OptionPricer(
        model=RoughHeston.create(hurst=0.1, adams_steps=200, **kwargs),
        method=OptionPricingMethod.COS,
    )
    heston = OptionPricer(model=Heston.create(**kwargs), method=OptionPricingMethod.COS)
    ttm = 0.1

    def skew(pricer: OptionPricer) -> float:
        lo = pricer.price(
            option_type=OptionType.PUT, strike=90.0, forward=100.0, ttm=ttm
        )
        hi = pricer.price(
            option_type=OptionType.CALL, strike=110.0, forward=100.0, ttm=ttm
        )
        return float(lo.black.iv) - float(hi.black.iv)

    assert skew(rough) > skew(heston)


def test_sample(rough_heston: RoughHeston) -> None:
    np.random.seed(42)
    paths = rough_heston.sample(2000, time_horizon=1.0, time_steps=100)
    assert paths.samples == 2000
    assert paths.time_steps == 100
    assert np.all(np.isfinite(paths.data))
    assert paths.data[0].std() == 0.0
