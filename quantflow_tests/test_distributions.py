import numpy as np
import pytest

from quantflow.utils.distributions import DoubleExponential


def test_double_exponential():
    d = DoubleExponential(decay=0.1)
    assert d.mean() == 0
    assert d.variance() == 200
    assert d.scale == 10


def test_double_exponential_samples():
    d = DoubleExponential(decay=0.1, kappa=2)
    samples = d.sample(10000)
    assert samples.shape == (10000,)
    assert samples.mean() == pytest.approx(d.mean(), rel=0.8)
    #
    d = DoubleExponential.from_moments(kappa=1)
    assert d.decay == pytest.approx(np.sqrt(2))
    assert d.mean() == 0
    assert d.variance() == pytest.approx(1)
    #
    d = DoubleExponential.from_moments(variance=2, kappa=2)
    assert d.mean() == 0
    assert d.variance() == pytest.approx(2)
    #
    d = DoubleExponential.from_moments(mean=-1, variance=2, kappa=2)
    assert d.mean() == -1
    assert d.variance() == pytest.approx(2)
