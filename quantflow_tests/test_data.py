from datetime import date
from typing import AsyncIterator
from unittest.mock import AsyncMock

import pandas as pd
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
    df = await fmp.prices("BTCUSD", frequency=fmp.freq.one_hour)
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
        assert df.shape[1] == 11
        assert isinstance(df.index, pd.DatetimeIndex)


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


async def test_fiscal_securities_builds_previous_month_filter() -> None:
    fd = FiscalData()
    fd.get_all = AsyncMock(return_value=[{"a": 1}])  # type: ignore[method-assign]
    df = await fd.securities(record_date=date(2024, 3, 15))
    fd.get_all.assert_awaited_once_with(
        "/v1/debt/mspd/mspd_table_3_market",
        {"filter": "record_date:eq:2024-02-29"},
    )
    assert len(df) == 1


async def test_fiscal_get_all_multi_page() -> None:
    fd = FiscalData()
    fd.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {
                "data": [{"id": 1}],
                "links": {"next": "/v1/debt/mspd/mspd_table_3_market?page=2"},
            },
            {"data": [{"id": 2}], "links": {"next": None}},
        ]
    )
    data = await fd.get_all("/v1/debt/mspd/mspd_table_3_market", {"a": "b"})
    assert data == [{"id": 1}, {"id": 2}]
    assert fd.get.await_count == 2


async def test_fiscal_get_all_single_page_without_links() -> None:
    fd = FiscalData()
    fd.get = AsyncMock(return_value={"data": [{"id": 7}]})  # type: ignore[method-assign]
    data = await fd.get_all("/v1/debt/mspd/mspd_table_3_market", {"a": "b"})
    assert data == [{"id": 7}]
    fd.get.assert_awaited_once()
