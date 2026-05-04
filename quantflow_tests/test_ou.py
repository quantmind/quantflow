import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from quantflow.sp.ou import GammaOU, Vasicek
from quantflow.sp.poisson import CompoundPoissonProcess
from quantflow.utils.distributions import Exponential
from quantflow_tests.utils import analytical_tests, characteristic_tests


@pytest.fixture
def vasicek() -> Vasicek:
    return Vasicek(kappa=5)


@pytest.fixture
def gamma_ou() -> GammaOU:
    return GammaOU.create(decay=10, kappa=5)


def test_marginal(gamma_ou: GammaOU) -> None:
    m = gamma_ou.marginal(1)
    assert m.mean() == 1


def test_sample(gamma_ou: GammaOU) -> None:
    paths = gamma_ou.sample(10, 1, 100)
    assert paths.t == 1
    assert paths.dt == 0.01


def test_gamma_ou_sample_mean_unbiased() -> None:
    # Use rate != stationary so the time-dependent mean is observable.
    # Pre-fix the off-by-one in _advance under-biased the empirical mean by
    # roughly kappa*dt; this test catches a ~2% bias at 20000 paths.
    np.random.seed(0)
    process = GammaOU(
        rate=0.5,
        kappa=2.0,
        bdlp=CompoundPoissonProcess[Exponential](
            intensity=2.0, jumps=Exponential(decay=10.0)
        ),
    )
    paths = process.sample(20000, time_horizon=5.0, time_steps=500)
    for t_idx, t in [(100, 1.0), (300, 3.0), (500, 5.0)]:
        assert paths.data[t_idx].mean() == pytest.approx(
            float(process.analytical_mean(t)), abs=0.005
        )


def test_gamma_ou_transient_moments() -> None:
    # rate != stationary mean so the time dependence is observable
    process = GammaOU(
        rate=0.5,
        kappa=2.0,
        bdlp=CompoundPoissonProcess[Exponential](
            intensity=2.0, jumps=Exponential(decay=10.0)
        ),
    )
    analytical_tests(process)
    # also check the limit converges to the stationary moments
    assert process.analytical_mean(100.0) == pytest.approx(0.2, abs=1e-6)
    assert process.analytical_variance(100.0) == pytest.approx(0.02, abs=1e-6)


def test_gamma_ou_integrated_log_laplace() -> None:
    process = GammaOU.create(rate=1.0, decay=10.0, kappa=1.0)
    t = 1.0
    assert process.integrated_log_laplace(t, 0.0) == pytest.approx(0.0)
    u = np.array([0.5, 1.0, 2.0])
    kappa = process.kappa
    beta = process.beta
    lam = process.intensity
    f = (1 - np.exp(-kappa * t)) / kappa
    expected = -u * process.rate * f + (lam / (u + beta * kappa)) * (
        beta * kappa * np.log((beta * kappa + u * f * kappa) / (beta * kappa))
        - u * kappa * t
    )
    np.testing.assert_allclose(
        process.integrated_log_laplace(t, u), expected, rtol=1e-12
    )


def test_vasicek(vasicek: Vasicek) -> None:
    m = vasicek.marginal(10)
    characteristic_tests(m)
    analytical_tests(vasicek)
    assert m.mean() == 1.0
    assert m.variance() == pytest.approx(0.1)
    assert m.mean_from_characteristic() == pytest.approx(1.0, 1e-3)
    assert m.std_from_characteristic() == pytest.approx(m.std(), 1e-3)


def test_vasicek_negative() -> None:
    # Vasicek admits negative initial values and mean levels
    process = Vasicek(rate=-0.5, kappa=2.0, theta=-0.2)
    m = process.marginal(1.0)
    expected_mean = -0.5 * np.exp(-2.0) + (-0.2) * (1 - np.exp(-2.0))
    assert m.mean() == pytest.approx(expected_mean)
    assert process.domain_range().lb == -np.inf


def test_vasicek_pdf_cdf_parity(vasicek: Vasicek) -> None:
    m = vasicek.marginal(1.0)
    pdf = m.pdf_from_characteristic(128)
    np.testing.assert_array_almost_equal(pdf.y, m.pdf(pdf.x), decimal=2)
    # analytical CDF is consistent with analytical PDF via numerical integration
    x = np.linspace(m.mean() - 6 * m.std(), m.mean() + 6 * m.std(), 4096)
    cdf_from_pdf = cumulative_trapezoid(m.pdf(x), x, initial=0) + m.cdf(x[0])
    np.testing.assert_array_almost_equal(m.cdf(x), cdf_from_pdf, decimal=4)
