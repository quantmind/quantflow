import numpy as np
import pytest

from quantflow.sp.bns import BNS
from quantflow_tests.utils import characteristic_tests


@pytest.fixture
def bns() -> BNS:
    return BNS.create(vol=0.5, decay=5, kappa=1, rho=0)


@pytest.fixture
def bns_leverage() -> BNS:
    return BNS.create(vol=0.5, decay=5, kappa=1, rho=-0.5)


def test_bns(bns: BNS) -> None:
    m = bns.marginal(1)
    characteristic_tests(m)
    assert bns.characteristic(1, 0) == 1
    assert m.mean_from_characteristic() == pytest.approx(0.0, abs=1e-6)
    # stationary variance = vol*vol = 0.25, so std on [0, 1] = 0.5
    assert m.std_from_characteristic() == pytest.approx(0.5, 1e-3)


def test_bns_horizon(bns: BNS) -> None:
    m = bns.marginal(2)
    # variance scales linearly with t when v0 equals stationary mean
    assert m.std_from_characteristic() == pytest.approx(2**0.5 * 0.5, 1e-3)


def test_bns_no_leverage_matches_integrated_laplace(bns: BNS) -> None:
    # at rho=0: E[exp(iu x_t)] = E[exp(-u^2/2 * int v_s ds)]
    # which is the GammaOU integrated Laplace transform evaluated at u^2/2
    u = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    cf_bns = bns.characteristic(1, u)
    cf_via_ou = np.exp(bns.variance_process.integrated_log_laplace(1, u * u / 2))
    np.testing.assert_allclose(cf_bns, cf_via_ou, atol=1e-12)


def test_bns_leverage(bns_leverage: BNS) -> None:
    m = bns_leverage.marginal(1)
    characteristic_tests(m)
    assert bns_leverage.characteristic(1, 0) == 1
    # E[x_t] = rho * kappa * t * intensity / beta = -0.5 * 1 * 1 * 1.25 / 5
    assert m.mean_from_characteristic() == pytest.approx(-0.125, 1e-3)


def test_bns_sample_moments(bns_leverage: BNS) -> None:
    np.random.seed(42)
    paths = bns_leverage.sample(5000, time_horizon=1.0, time_steps=200)
    assert paths.data.shape == (201, 5000)
    # Compare empirical moments at T=1 against the characteristic-function moments
    m = bns_leverage.marginal(1)
    terminal = paths.data[-1]
    assert terminal.mean() == pytest.approx(
        float(m.mean_from_characteristic()), abs=0.02
    )
    assert terminal.std() == pytest.approx(float(m.std_from_characteristic()), abs=0.02)
