import pytest

from quantflow.data.fmp import FMP

pytestmark = pytest.mark.skipif(FMP().key is None, reason="No FMP API key found")


@pytest.fixture
def fmp() -> FMP:
    return FMP()


def test_client(fmp: FMP) -> None:
    assert fmp.url
    assert fmp.key


async def test_historical(fmp: FMP) -> None:
    df = await fmp.prices("BTCUSD")
    assert df["close"] is not None
