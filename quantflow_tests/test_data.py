from typing import AsyncIterator

import pytest
from aiohttp.client_exceptions import ClientError

from quantflow.data.fed import FederalReserve
from quantflow.data.fiscal_data import FiscalData
from quantflow.data.fmp import FMP

pytestmark = pytest.mark.skipif(not FMP().key, reason="No FMP API key found")


@pytest.fixture
async def fmp() -> AsyncIterator[FMP]:
    async with FMP() as fmp:
        yield fmp


def test_client(fmp: FMP) -> None:
    assert fmp.url
    assert fmp.key


async def test_historical(fmp: FMP) -> None:
    df = await fmp.prices("BTCUSD", fmp.freq.one_hour)
    assert df["close"] is not None


async def test_dividends(fmp: FMP) -> None:
    data = await fmp.dividends()
    assert data is not None


async def test_fed_yc() -> None:
    try:
        async with FederalReserve() as fed:
            df = await fed.yield_curves()
            assert df is not None
            assert df.shape[0] > 0
            assert df.shape[1] == 12
    except (ConnectionError, ClientError) as e:
        pytest.skip(f"Skipping test_fed due to network issue: {e}")


async def test_fed_rates() -> None:
    try:
        async with FederalReserve() as fed:
            df = await fed.ref_rates()
            assert df is not None
            assert df.shape[0] > 0
            assert df.shape[1] == 2
    except (ConnectionError, ClientError) as e:
        pytest.skip(f"Skipping test_fed due to network issue: {e}")


async def __test_fiscal_data() -> None:
    try:
        async with FiscalData() as fd:
            df = await fd.securities()
            assert df is not None
            assert df.shape[0] > 0
            assert df.shape[1] == 2
    except (ConnectionError, ClientError) as e:
        pytest.skip(f"Skipping test_fed due to network issue: {e}")
