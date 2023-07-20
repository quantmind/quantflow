import pytest

from quantflow.options.pricer import OptionPricer
from quantflow.sp.heston import HestonJ
from tests.utils import has_plotly


@pytest.fixture
def pricer() -> OptionPricer[HestonJ]:
    return OptionPricer(HestonJ.create(vol=0.5, kappa=1, sigma=0.8, rho=0))


@pytest.mark.skipif(not has_plotly, reason="Plotly not installed")
def test_plot_surface(pricer: OptionPricer):
    fig = pricer.plot3d()
    surface = fig.data[0]
    assert surface.x is not None
    assert surface.y is not None
    assert surface.z is not None
