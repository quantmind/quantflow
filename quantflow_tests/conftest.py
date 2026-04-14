import json
from pathlib import Path

import dotenv
import pytest

from quantflow.options.surface import VolSurfaceInputs, surface_from_inputs

dotenv.load_dotenv()

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())


def load_fixture_dict(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def vol_surface():
    inputs = load_fixture_dict("volsurface.json")
    return surface_from_inputs(VolSurfaceInputs(**inputs))
