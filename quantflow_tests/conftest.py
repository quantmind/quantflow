import dotenv
import pytest

from quantflow.options.surface import VolSurfaceInputs, surface_from_inputs
from quantflow_tests.utils import load_fixture_dict

dotenv.load_dotenv()


@pytest.fixture
def vol_surface():
    inputs = load_fixture_dict("volsurface.json")
    return surface_from_inputs(VolSurfaceInputs(**inputs))
