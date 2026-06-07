from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pandas as pd

from quantflow.data.fmp import FMP, nice_sector_performance, summary_sector_performance


def test_freq_crate_and_join_and_params() -> None:
    assert FMP.freq.crate(None) == FMP.freq.DAILY
    assert FMP.freq.crate("1hour") == FMP.freq.ONE_HOUR
    assert FMP.freq.crate("bad") == FMP.freq.DAILY

    fmp = FMP(key="k")
    assert fmp.join("AAPL", "MSFT") == "AAPL,MSFT"
    assert fmp.params({"a": 1}) == {"params": {"a": 1, "apikey": "k"}}


async def test_prices_daily_and_intraday_paths() -> None:
    fmp = FMP(key="k")
    fmp.get_path = AsyncMock(return_value=[{"date": "2024-01-01", "close": 100}])  # type: ignore[method-assign]
    df = await fmp.prices("AAPL", frequency=None, convert_to_date=True)
    assert isinstance(df, pd.DataFrame)
    assert str(df["date"].dtype).startswith("datetime64")
    fmp.get_path.assert_awaited_with(
        "historical-price-eod/full",
        params={"symbol": "AAPL"},
    )

    fmp.get_path = AsyncMock(return_value={"historical": [{"date": "2024-01-01"}]})  # type: ignore[method-assign]
    await fmp.prices("AAPL", frequency="1hour")
    fmp.get_path.assert_awaited_with(
        "historical-chart/1hour",
        params={"frequency": "1hour", "symbol": "AAPL"},
    )


async def test_sector_performance_summary_and_timeseries() -> None:
    fmp = FMP(key="k")
    fmp.get_path = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [{"sector": "Tech", "changesPercentage": "1.23%"}],
            [
                {"date": "2024-01-01", "technologyChangesPercentage": 1.0},
                {"date": "2024-01-02", "technologyChangesPercentage": 2.0},
            ],
        ]
    )
    snapshot = await fmp.sector_performance()
    assert isinstance(snapshot, dict)
    assert str(snapshot["Tech"]) == "1.23"

    summary = await fmp.sector_performance(from_date=date(2024, 1, 1), summary=True)
    assert isinstance(summary, dict)
    assert str(summary["Technology"]) == "3.02"


def test_sector_helpers() -> None:
    nice = dict(
        nice_sector_performance(
            {"date": "2024-01-01", "consumerStaplesChangesPercentage": 1.5}
        )
    )
    assert nice["date"] == date(2024, 1, 1)
    assert nice["Consumer Staples"] == 1.5

    summary = summary_sector_performance(
        [
            {"date": date(2024, 1, 1), "Tech": 1.0},
            {"date": date(2024, 1, 2), "Tech": 2.0},
        ]
    )
    assert str(summary["Tech"]) == "3.02"
