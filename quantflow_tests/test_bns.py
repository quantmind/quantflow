import numpy as np
import pytest

from quantflow.sp.bns import BNS, BNS2
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


@pytest.fixture
def bns2() -> BNS2:
    # Fast-mean-reverting plus slow-mean-reverting factor.
    return BNS2(
        bns1=BNS.create(vol=0.3, decay=5, kappa=3, rho=-0.3),
        bns2=BNS.create(vol=0.4, decay=5, kappa=1, rho=-0.5),
        weight=0.4,
    )


def test_bns2_characteristic(bns2: BNS2) -> None:
    assert bns2.bns1.variance_process.kappa > bns2.bns2.variance_process.kappa
    assert bns2.characteristic(1, 0) == 1
    m = bns2.marginal(1)
    characteristic_tests(m)
    # The leverage term is independent of the diffusion weighting, so the BNS2
    # mean is just the sum of the per-factor leverage means.
    m1 = bns2.bns1.marginal(1)
    m2 = bns2.bns2.marginal(1)
    expected_mean = float(m1.mean_from_characteristic() + m2.mean_from_characteristic())
    assert m.mean_from_characteristic() == pytest.approx(expected_mean, abs=1e-6)


def test_bns2_collapses_to_single_factor() -> None:
    # With weight = 1 the second factor only contributes leverage, and removing
    # that leverage (rho2 = 0) reduces BNS2 to its first-factor BNS.
    bns1 = BNS.create(vol=0.4, decay=5, kappa=2, rho=-0.4)
    bns2 = BNS.create(vol=0.5, decay=3, kappa=1, rho=0.0)
    pair = BNS2(bns1=bns1, bns2=bns2, weight=1.0)
    u = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    np.testing.assert_allclose(
        pair.characteristic(1, u), bns1.characteristic(1, u), atol=1e-12
    )


def test_bns2_sample_moments(bns2: BNS2) -> None:
    np.random.seed(42)
    paths = bns2.sample(5000, time_horizon=1.0, time_steps=200)
    assert paths.data.shape == (201, 5000)
    m = bns2.marginal(1)
    terminal = paths.data[-1]
    assert terminal.mean() == pytest.approx(
        float(m.mean_from_characteristic()), abs=0.03
    )
    assert terminal.std() == pytest.approx(float(m.std_from_characteristic()), abs=0.03)
