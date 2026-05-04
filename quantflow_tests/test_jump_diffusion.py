import numpy as np
import pytest
from scipy.stats import norm, poisson

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


def merton_analytical_pdf(
    merton: JumpDiffusion[Normal], t: float, x: np.ndarray, max_n: int = 60
) -> np.ndarray:
    # Merton model: x_t = sigma*W_t + sum_{i=1}^{N_t} J_i with J_i ~ N(mu, sigma_J^2),
    # density is the Poisson-weighted mixture of Gaussians.
    sigma = merton.diffusion.sigma
    lam = merton.jumps.intensity
    mu_j = merton.jumps.jumps.mu
    sigma_j = merton.jumps.jumps.sigma
    pdf = np.zeros_like(x, dtype=float)
    for n in range(max_n):
        var = sigma * sigma * t + n * sigma_j * sigma_j
        pdf += poisson.pmf(n, lam * t) * norm.pdf(x, loc=n * mu_j, scale=np.sqrt(var))
    return pdf


def test_pdf_vs_characteristic(merton: JumpDiffusion[Normal]) -> None:
    t = 1.0
    m = merton.marginal(t)
    pdf = m.pdf_from_characteristic(256)
    analytical = merton_analytical_pdf(merton, t, np.asarray(pdf.x))
    # restrict to the bulk where the analytical density is non-negligible:
    # the FFT-based reconstruction has tail aliasing where pdf ~ 0.
    mask = analytical > 1e-3
    np.testing.assert_array_almost_equal(pdf.y[mask], analytical[mask], decimal=2)
