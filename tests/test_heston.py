import pytest

from quantflow.sp.heston import Heston
from tests.utils import characteristic_tests


@pytest.fixture
def heston() -> Heston:
    return Heston.create(vol=0.5, kappa=1, sigma=0.5, rho=0)


def test_characteristic(heston: Heston) -> None:
    assert heston.variance_process.is_positive is True
    assert heston.characteristic(1, 0) == 1
    m = heston.marginal(1)
    characteristic_tests(m)
    assert m.mean() == 0.0
    assert pytest.approx(m.std()) == 0.5
