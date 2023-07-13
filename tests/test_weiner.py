import numpy as np
import pytest

from quantflow.sp.weiner import WeinerProcess
from quantflow.utils.paths import Paths


@pytest.fixture
def weiner() -> WeinerProcess:
    return WeinerProcess(sigma=0.5)


def test_characteristic(weiner: WeinerProcess) -> None:
    assert weiner.characteristic(1, 0) == 1
    assert weiner.convexity_correction(2) == 0.25
    marginal = weiner.marginal(1)
    assert marginal.mean() == 0
    assert marginal.mean_from_characteristic() == 0
    assert marginal.std() == 0.5
    assert marginal.std_from_characteristic() == pytest.approx(0.5)
    assert marginal.variance_from_characteristic() == pytest.approx(0.25)
    df = marginal.characteristic_df(128)
    assert len(df.columns) == 3


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


def test_normal_draws1() -> None:
    paths = Paths.normal_draws(1, 1, 1000)
    assert paths.samples == 1
    assert paths.time_steps == 1000
    paths = Paths.normal_draws(1, 1, 1000, antithetic_variates=False)
    assert paths.samples == 1
    assert paths.time_steps == 1000


def test_support(weiner: WeinerProcess) -> None:
    m = weiner.marginal(0.01)
    pdf = m.pdf_from_characteristic(32)
    assert len(pdf.x) == 32


def test_fft_v_frft(weiner: WeinerProcess) -> None:
    m = weiner.marginal(1)
    pdf1 = m.pdf_from_characteristic(128, max_frequency=10)
    pdf2 = m.pdf_from_characteristic(128, use_fft=True, max_frequency=200)
    y = np.interp(pdf1.x[10:-10], pdf2.x, pdf2.y)
    assert np.allclose(y, pdf1.y[10:-10], 1e-2)
    #
    # TODO: simpson rule seems to fail for FFT
    # pdf1 = m.pdf_from_characteristic(128, max_frequency=10, simpson_rule=True)
    # pdf2 = m.pdf_from_characteristic(
    #    128, use_fft=True, max_frequency=200, simpson_rule=True
    # )
    # y = np.interp(pdf1.x[10:-10], pdf2.x, pdf2.y)
    # assert np.allclose(y, pdf1.y[10:-10], 1e-2)
