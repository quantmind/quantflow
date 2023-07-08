import math

import pytest

from quantflow.sp.poisson import PoissonProcess


@pytest.fixture
def poisson() -> PoissonProcess:
    return PoissonProcess(rate=2)


def test_characteristic(poisson: PoissonProcess) -> None:
    m1 = poisson.marginal(1)
    m2 = poisson.marginal(2)
    assert poisson.characteristic(1, 0) == 1
    assert m1.mean() == 2
    assert pytest.approx(m1.mean_from_characteristic(), 0.001) == 2
    assert pytest.approx(m2.mean_from_characteristic(), 0.001) == 4
    assert pytest.approx(m1.std()) == math.sqrt(2)
    assert pytest.approx(m1.variance_from_characteristic(), 0.001) == 2
    assert pytest.approx(m2.variance_from_characteristic(), 0.001) == 4


def test_sampling(poisson: PoissonProcess) -> None:
    paths = poisson.sample(1000, time_horizon=1, time_steps=1000)
    mean = paths.mean()
    assert mean[0] == 0
    std = paths.std()
    assert std[0] == 0
