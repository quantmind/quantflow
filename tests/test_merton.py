import pytest

from quantflow.sp.jump_diffusion import Merton


@pytest.fixture
def merton() -> Merton:
    return Merton.create()


def test_characteristic(merton: Merton) -> None:
    m = merton.marginal(1)
    assert m.mean() == 0.0
    assert pytest.approx(m.std()) == 0.5
