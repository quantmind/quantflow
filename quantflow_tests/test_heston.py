import pytest

from quantflow.sp.heston import Heston, HestonJ
from quantflow.utils.distributions import DoubleExponential
from quantflow_tests.utils import characteristic_tests


@pytest.fixture
def heston() -> Heston:
    return Heston.create(vol=0.5, kappa=1, sigma=0.5, rho=0)


@pytest.fixture
def heston_jumps() -> HestonJ[DoubleExponential]:
    return HestonJ.create(
        DoubleExponential,
        vol=0.5,
        kappa=1,
        sigma=0.5,
        jump_intensity=50,
        jump_fraction=0.3,
    )


def test_characteristic(heston: Heston) -> None:
    assert heston.variance_process.is_positive is True
    assert heston.characteristic(1, 0) == 1
    m = heston.marginal(1)
    characteristic_tests(m)
    assert m.mean() == 0.0
    assert pytest.approx(m.std()) == 0.5


def test_heston_jumps_characteristic(heston_jumps: HestonJ) -> None:
    assert heston_jumps.variance_process.is_positive is True
    m = heston_jumps.marginal(1)
    characteristic_tests(m)
    assert m.mean() == 0.0
    assert m.std() == pytest.approx(0.5)
