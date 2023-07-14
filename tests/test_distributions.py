import numpy as np
import pytest

from quantflow.utils.distributions import DoubleExponential


def test_duuble_exponential():
    d = DoubleExponential(decay=0.1)
    assert d.mean() == 0
    assert d.variance() == 100
    assert d.scale == 10
    assert d.scale2_up == 50
    assert d.scale2_down == 50
    assert d.scale_up == np.sqrt(50)
    assert d.scale_up == np.sqrt(50)
    #
    d = DoubleExponential(decay=0.1, k=2)
    assert d.mean() == pytest.approx(np.sqrt(80) - np.sqrt(20))
    assert d.variance() == 100
    assert d.scale == 10
    assert d.scale2_up == 80
    assert d.scale2_down == 20
