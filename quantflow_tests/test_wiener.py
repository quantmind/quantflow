import numpy as np
import pytest

from quantflow.sp.wiener import WienerProcess
from quantflow_tests.utils import characteristic_tests


@pytest.fixture
def wiener() -> WienerProcess:
    return WienerProcess(sigma=0.5)


def test_characteristic(wiener: WienerProcess) -> None:
    assert wiener.characteristic(1, 0) == 1
    assert wiener.convexity_correction(2) == 0.25
    marginal = wiener.marginal(1)
    characteristic_tests(marginal)
    assert marginal.mean() == 0
    assert marginal.mean_from_characteristic() == 0
    assert marginal.std() == 0.5
    assert marginal.std_from_characteristic() == pytest.approx(0.5)
    assert marginal.variance_from_characteristic() == pytest.approx(0.25)
    df = marginal.characteristic_df(128)
    assert len(df.columns) == 3


def test_sampling(wiener: WienerProcess) -> None:
    paths = wiener.sample(1000, time_horizon=1, time_steps=1000)
    mean = paths.mean()
    assert mean[0] == 0
    std = paths.std()
    assert std[0] == 0


def test_support(wiener: WienerProcess) -> None:
    m = wiener.marginal(0.01)
    pdf = m.pdf_from_characteristic(32)
    assert len(pdf.x) == 32


def test_fft_v_frft(wiener: WienerProcess) -> None:
    m = wiener.marginal(1)
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
