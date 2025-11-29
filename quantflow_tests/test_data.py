from typing import AsyncIterator

import pytest

from quantflow.data.fed import FederalReserve
from quantflow.data.fiscal_data import FiscalData
from quantflow.data.fmp import FMP
from quantflow_tests.utils import skip_network_issue

skip_fmp = not FMP().key


@pytest.fixture
async def fmp() -> AsyncIterator[FMP]:
    async with FMP() as fmp:
        yield fmp


@pytest.mark.skipif(skip_fmp, reason="No FMP API key found")
def test_client(fmp: FMP) -> None:
    assert fmp.url
    assert fmp.key


@pytest.mark.skipif(skip_fmp, reason="No FMP API key found")
async def test_historical(fmp: FMP) -> None:
    df = await fmp.prices("BTCUSD", fmp.freq.one_hour)
    assert df["close"] is not None


@pytest.mark.skipif(skip_fmp, reason="No FMP API key found")
async def test_dividends(fmp: FMP) -> None:
    data = await fmp.dividends()
    assert data is not None


@skip_network_issue
async def test_fed_yc() -> None:
    async with FederalReserve() as fed:
        df = await fed.yield_curves()
        assert df is not None
        assert df.shape[0] > 0
        assert df.shape[1] == 12


@skip_network_issue
async def test_fed_rates() -> None:
    async with FederalReserve() as fed:
        df = await fed.ref_rates()
        assert df is not None
        assert df.shape[0] > 0
        assert df.shape[1] == 2


@skip_network_issue
async def __test_fiscal_data() -> None:
    async with FiscalData() as fd:
        df = await fd.securities()
        assert df is not None
        assert df.shape[0] > 0
        assert df.shape[1] == 2
