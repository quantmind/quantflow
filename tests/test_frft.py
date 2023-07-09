import numpy as np
import pytest
from scipy.optimize import Bounds
from quantflow.utils.transforms import frft, Transform


@pytest.fixture
def x():
    t = np.linspace(-4 * np.pi, 4 * np.pi, 64)
    return (
        np.sin(2 * np.pi * 40 * t)
        + np.sin(2 * np.pi * 20 * t)
        + np.sin(2 * np.pi * 10 * t)
    )


def test_frft(x):
    t = frft.calculate(x, 0.01)
    assert t.n == 64


def test_transform_positive_domain():
    n = 10
    t = Transform(n, domain_range=Bounds(0, np.inf))
    assert t.n == n
    x = t.space_domain(1)
    assert len(x) == n
    np.testing.assert_almost_equal(x, np.linspace(0, n - 1, n))
