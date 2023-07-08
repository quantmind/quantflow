import numpy as np
import pytest

from quantflow.sp.cir import CIR, SamplingAlgorithm


@pytest.fixture
def cir_neg() -> CIR:
    return CIR(kappa=1, sigma=2, sample_algo=SamplingAlgorithm.euler)


def test_cir_neg(cir_neg: CIR) -> None:
    assert cir_neg.is_positive is False
    assert cir_neg.sigma2 == 4
    assert cir_neg.marginal(1).mean() == 1.0
    assert cir_neg.marginal(1).mean_from_characteristic() == pytest.approx(1.0, 1e-3)


def test_cir_neg_sampling(cir_neg: CIR) -> None:
    paths = cir_neg.sample(10, time_horizon=1, time_steps=1000)
    assert paths.samples == 10
    assert paths.time_steps == 1000
    assert paths.dt == 0.001
    assert np.all(paths.data == paths.data)
