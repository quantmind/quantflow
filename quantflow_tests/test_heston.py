import numpy as np
import pytest

from quantflow.sp.heston import DoubleHeston, DoubleHestonJ, Heston, HestonJ
from quantflow.utils.distributions import DoubleExponential
from quantflow_tests.utils import characteristic_tests


@pytest.fixture
def heston() -> Heston:
    return Heston.create(vol=0.5, kappa=1, sigma=0.5, rho=0)


@pytest.fixture
def heston_jumps() -> HestonJ[DoubleExponential]:
    return HestonJ.create(
        DoubleExponential,
        vol=0.5,
        kappa=1,
        sigma=0.5,
        jump_intensity=50,
        jump_fraction=0.3,
    )


def test_characteristic(heston: Heston) -> None:
    assert heston.variance_process.is_positive is True
    assert heston.characteristic(1, 0) == 1
    m = heston.marginal(1)
    characteristic_tests(m)
    assert m.mean() == 0.0
    assert pytest.approx(m.std()) == 0.5


def test_heston_jumps_characteristic(heston_jumps: HestonJ) -> None:
    assert heston_jumps.variance_process.is_positive is True
    m = heston_jumps.marginal(1)
    characteristic_tests(m)
    assert m.mean() == 0.0
    assert m.std() == pytest.approx(0.5)


@pytest.fixture
def double_heston() -> DoubleHeston:
    return DoubleHeston(
        heston1=Heston.create(vol=0.3, kappa=3, sigma=0.5, rho=-0.3),
        heston2=Heston.create(vol=0.4, kappa=1, sigma=0.5, rho=-0.5),
    )


@pytest.fixture
def double_heston_jumps() -> DoubleHestonJ[DoubleExponential]:
    return DoubleHestonJ(
        heston1=HestonJ.create(
            DoubleExponential,
            vol=0.3,
            kappa=3,
            sigma=0.5,
            rho=-0.3,
            jump_intensity=50,
            jump_fraction=0.2,
        ),
        heston2=Heston.create(vol=0.4, kappa=1, sigma=0.5, rho=-0.5),
    )


def test_double_heston_characteristic(double_heston: DoubleHeston) -> None:
    assert (
        double_heston.heston1.variance_process.kappa
        > double_heston.heston2.variance_process.kappa
    )
    assert double_heston.characteristic(1, 0) == 1
    m = double_heston.marginal(1)
    characteristic_tests(m)
    assert m.mean() == pytest.approx(0.0, abs=1e-6)
    assert m.std() == pytest.approx(0.5, rel=0.01)


def test_double_heston_jumps_characteristic(
    double_heston_jumps: DoubleHestonJ[DoubleExponential],
) -> None:
    assert double_heston_jumps.characteristic(1, 0) == 1
    m = double_heston_jumps.marginal(1)
    characteristic_tests(m)
    assert m.mean() == pytest.approx(0.0, abs=1e-6)
    assert m.std() == pytest.approx(0.5, rel=0.05)


def test_heston_jumps_sample_includes_jumps(
    heston_jumps: HestonJ[DoubleExponential],
) -> None:
    np.random.seed(42)
    paths = heston_jumps.sample(5000, time_horizon=1.0, time_steps=200)
    # If jumps were dropped, std would collapse to sqrt(1 - jump_fraction) * 0.5
    # ~ 0.418, well below 0.5
    assert paths.data[-1].std() == pytest.approx(0.5, abs=0.03)


def test_double_heston_jumps_sample_includes_jumps(
    double_heston_jumps: DoubleHestonJ[DoubleExponential],
) -> None:
    np.random.seed(42)
    paths = double_heston_jumps.sample(5000, time_horizon=1.0, time_steps=200)
    assert paths.data[-1].std() == pytest.approx(0.5, abs=0.03)
