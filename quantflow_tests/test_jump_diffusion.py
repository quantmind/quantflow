import pytest

from quantflow.sp.jump_diffusion import JumpDiffusion
from quantflow.utils.distributions import Normal


@pytest.fixture
def merton() -> JumpDiffusion[Normal]:
    return JumpDiffusion.create(Normal, jump_fraction=0.8)


def test_characteristic(merton: JumpDiffusion[Normal]) -> None:
    m = merton.marginal(1)
    assert m.mean() == 0
    assert pytest.approx(m.std()) == pytest.approx(0.5, 1.0e-3)
    pdf = m.pdf_from_characteristic(128)
    assert pdf.x[0] < 0
    assert pdf.x[-1] > 0
    assert -pdf.x[0] != pdf.x[-1]


def test_sampling(merton: JumpDiffusion[Normal]) -> None:
    paths = merton.sample(1000, time_horizon=1, time_steps=1000)
    mean = paths.mean()
    assert mean[0] == 0
    std = paths.std()
    assert std[0] == 0
