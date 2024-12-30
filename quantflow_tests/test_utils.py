import numpy as np

from quantflow.ta.paths import Paths
from quantflow.utils.numbers import round_to_step, to_decimal


def test_round_to_step():
    assert str(round_to_step(1.234, 0.1)) == "1.2"
    assert str(round_to_step(1.234, 0.01)) == "1.23"
    assert str(round_to_step(1.236, 0.01)) == "1.24"
    assert str(round_to_step(1.1, 0.01)) == "1.10"
    assert str(round_to_step(1.1, 0.001)) == "1.100"
    assert str(round_to_step(2, 0.001)) == "2.000"
    assert str(round_to_step(to_decimal("2.00000000000"), 0.001)) == "2.000"


def test_normal_draws() -> None:
    paths = Paths.normal_draws(100, 1, 1000)
    assert paths.samples == 100
    assert paths.time_steps == 1000
    m = paths.mean()
    np.testing.assert_array_almost_equal(m, 0)
    paths = Paths.normal_draws(100, 1, 1000, antithetic_variates=False)
    assert np.abs(paths.mean().mean()) > np.abs(m.mean())


def test_normal_draws1() -> None:
    paths = Paths.normal_draws(1, 1, 1000)
    assert paths.samples == 1
    assert paths.time_steps == 1000
    paths = Paths.normal_draws(1, 1, 1000, antithetic_variates=False)
    assert paths.samples == 1
    assert paths.time_steps == 1000


def test_path_stats() -> None:
    paths = Paths.normal_draws(paths=2, time_steps=1000)
    assert paths.paths_mean().shape == (2,)
    assert paths.paths_std(scaled=True).shape == (2,)
    assert paths.paths_var(scaled=False).shape == (2,)
