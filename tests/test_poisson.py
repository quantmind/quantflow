import math

import numpy as np
import pytest

from quantflow.sp.dsp import DSP
from quantflow.sp.poisson import CompoundPoissonProcess, PoissonProcess
from quantflow.utils.distributions import Exponential


@pytest.fixture
def poisson() -> PoissonProcess:
    return PoissonProcess(intensity=2)


@pytest.fixture
def comp() -> CompoundPoissonProcess:
    return CompoundPoissonProcess(intensity=2, jumps=Exponential(decay=10))


@pytest.fixture
def dsp() -> DSP:
    return DSP()


def test_characteristic(poisson: PoissonProcess) -> None:
    m1 = poisson.marginal(1)
    m2 = poisson.marginal(2)
    assert poisson.characteristic(1, 0) == 1
    assert m1.mean() == 2
    assert pytest.approx(m1.mean_from_characteristic(), 0.001) == 2
    assert pytest.approx(m2.mean_from_characteristic(), 0.001) == 4
    assert pytest.approx(m1.std()) == math.sqrt(2)
    assert pytest.approx(m1.variance_from_characteristic(), 0.001) == 2
    assert pytest.approx(m2.variance_from_characteristic(), 0.001) == 4


def test_pdf(poisson: PoissonProcess) -> None:
    m = poisson.marginal(1)
    x = 1.0 * np.arange(10)
    # m.pdf(x)
    c_pdf = m.pdf_from_characteristic(x)
    np.testing.assert_almost_equal(x[:9], c_pdf.x)
    # TODO: fix this
    # np.testing.assert_almost_equal(pdf, c_pdf.y[:10])


def test_sampling(poisson: PoissonProcess) -> None:
    paths = poisson.sample(1000, time_horizon=1, time_steps=1000)
    mean = paths.mean()
    assert mean[0] == 0
    std = paths.std()
    assert std[0] == 0


def test_dsp_sample(dsp: DSP):
    paths = dsp.sample(1000, time_horizon=1, time_steps=1000)
    mean = paths.mean()
    assert mean[0] == 0
    std = paths.std()
    assert std[0] == 0
