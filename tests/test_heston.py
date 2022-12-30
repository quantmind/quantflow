import pytest

from quantflow.sp.heston import Heston


@pytest.fixture
def heston() -> Heston:
    return Heston.create(vol=0.5, kappa=1, sigma=0.5, rho=0)


def test_characteristic(heston: Heston) -> None:
    assert heston.variance_process.is_positive is True
    assert heston.characteristic(1, 0) == 1
    assert heston.mean(1) == 0.0
    assert pytest.approx(heston.std(1)) == 0.5
