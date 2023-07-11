import pytest

from quantflow.data.fmp import FMP


@pytest.fixture
def fmp() -> FMP:
    return FMP()


def test_client(fmp: FMP) -> None:
    assert fmp.url
