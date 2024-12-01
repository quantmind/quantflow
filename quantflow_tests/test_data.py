from typing import AsyncIterator

import pytest

try:
    from quantflow.data.fmp import FMP
except ImportError:
    FMP = None  # type: ignore

pytestmark = pytest.mark.skipif(
    FMP is None or not FMP().key, reason="No FMP API key found"
)


@pytest.fixture
async def fmp() -> AsyncIterator[FMP]:
    async with FMP() as fmp:
        yield fmp


def test_client(fmp: FMP) -> None:
    assert fmp.url
    assert fmp.key


async def test_historical(fmp: FMP) -> None:
    df = await fmp.prices("BTCUSD")
    assert df["close"] is not None


async def test_dividends(fmp: FMP) -> None:
    data = await fmp.dividends()
    assert data is not None
