import numpy as np
import pytest

from quantflow.sp.weiner import WeinerProcess
from quantflow.utils.paths import Paths


@pytest.fixture
def weiner() -> WeinerProcess:
    return WeinerProcess(sigma=0.5)


def test_characteristic(weiner: WeinerProcess) -> None:
    assert weiner.characteristic(1, 0) == 1
    marginal = weiner.marginal(1, 0)
    assert marginal.mean() == 0
    assert marginal.mean_from_characteristic() == 0
    assert marginal.std() == 0.5
    assert marginal.variance_from_characteristic() == pytest.approx(0.25)


def test_sampling(weiner: WeinerProcess) -> None:
    paths = weiner.sample(1000, time_horizon=1, time_steps=1000)
    mean = paths.mean()
    assert mean[0] == 0
    std = paths.std()
    assert std[0] == 0


def test_normal_draws() -> None:
    paths = Paths.normal_draws(100, 1, 1000)
    assert paths.samples == 100
    assert paths.time_steps == 1000
    m = paths.mean()
    np.testing.assert_array_almost_equal(m, 0)
    paths = Paths.normal_draws(100, 1, 1000, antithetic_variates=False)
    assert np.abs(paths.mean().mean()) > np.abs(m.mean())