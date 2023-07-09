import pytest

from quantflow.sp.ou import GammaOU


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
