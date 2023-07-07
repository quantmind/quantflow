import math

import pytest

from quantflow.sp.poisson import PoissonProcess


@pytest.fixture
def poisson() -> PoissonProcess:
    return PoissonProcess(rate=2)


def test_characteristic(poisson: PoissonProcess) -> None:
    assert poisson.characteristic(1, 0) == 1
    assert poisson.mean(1) == 2
    assert pytest.approx(poisson.mean_from_characteristic(1), 0.001) == 2
    assert pytest.approx(poisson.mean_from_characteristic(2), 0.001) == 4
    assert pytest.approx(poisson.std(1)) == math.sqrt(2)
    assert pytest.approx(poisson.variance_from_characteristic(1), 0.001) == 2
    assert pytest.approx(poisson.variance_from_characteristic(2), 0.001) == 4
