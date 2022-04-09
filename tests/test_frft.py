import numpy as np
import pytest

from quantflow.utils.frft import frft


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
