import pytest

from quantflow.sp.jump_diffusion import Merton


@pytest.fixture
def merton() -> Merton:
    return Merton.create(diffusion_percentage=0.2, jump_skew=-0.1)


def test_characteristic(merton: Merton) -> None:
    m = merton.marginal(1)
    assert m.mean() < 0
    assert pytest.approx(m.std()) == pytest.approx(0.5, 1.0e-3)
    pdf = m.pdf_from_characteristic(128)
    assert pdf.x[0] < 0
    assert pdf.x[-1] > 0
    assert -pdf.x[0] != pdf.x[-1]


def test_sampling(merton: Merton) -> None:
    paths = merton.sample(1000, time_horizon=1, time_steps=1000)
    mean = paths.mean()
    assert mean[0] == 0
    std = paths.std()
    assert std[0] == 0
