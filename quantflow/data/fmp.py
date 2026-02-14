import os
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any, Iterator, cast

import inflection
import pandas as pd
from fluid.utils.data import compact_dict
from fluid.utils.http_client import AioHttpClient
from typing_extensions import Annotated, Doc

from quantflow.utils.dates import to_date_iso
from quantflow.utils.numbers import to_decimal


@dataclass
class FMP(AioHttpClient):
    """Financial Modeling Prep API client"""

    url: str = "https://financialmodelingprep.com/stable"
    key: str = field(default_factory=lambda: os.environ.get("FMP_API_KEY", ""))

    class freq(StrEnum):
        """FMP historical frequencies"""

        one_min = "1min"
        five_min = "5min"
        fifteen_min = "15min"
        thirty_min = "30min"
        one_hour = "1hour"
        four_hour = "4hour"
        daily = ""

    async def market_risk_premium(self) -> list[dict]:
        """Market risk premium"""
        return await self.get_path("market-risk-premium")

    async def stocks(self, **kw: Any) -> list[dict]:
        return await self.get_path("stock-list", **kw)

    async def etfs(self, **kw: Any) -> list[dict]:
        return await self.get_path("etf-list", **kw)

    async def indices(self, **kw: Any) -> list[dict]:
        """Retrieve a comprehensive list of stock market indexes
        across global exchanges"""
        return await self.get_path("index-list", **kw)

    async def profile(
        self,
        *tickers: Annotated[str, Doc("One or more ticker symbols")],
        **kw: Any,
    ) -> list[dict]:
        """Company profile - minute"""
        return await self.get_path(f"profile/{self.join(*tickers)}", **kw)

    async def quote(
        self,
        *tickers: Annotated[str, Doc("One or more ticker symbols")],
        **kw: Any,
    ) -> list[dict]:
        """Company quote - real time"""
        return await self.get_path(f"quote/{self.join(*tickers)}", **kw)

    # calendars

    async def dividends(
        self,
        from_date: Annotated[
            str | date | None, Doc("Start date for dividend calendar")
        ] = None,
        to_date: Annotated[
            str | date | None, Doc("End date for dividend calendar")
        ] = None,
        **kw: Any,
    ) -> list[dict]:
        """Dividend calendar"""
        params = compact_dict(
            {"from": to_date_iso(from_date), "to": to_date_iso(to_date)},
        )
        return await self.get_path("dividends-calendar", params=params, **kw)

    # Executives

    async def executives(
        self,
        ticker: Annotated[str, Doc("Ticker symbol")],
        **kw: Any,
    ) -> list[dict]:
        """Company quote - real time"""
        return await self.get_path(f"key-executives/{ticker}", **kw)

    async def insider_trading(
        self,
        ticker: Annotated[str, Doc("Ticker symbol")],
        **kw: Any,
    ) -> list[dict]:
        """Company Insider Trading"""
        return await self.get_path(
            "insider-trading", **self.params(dict(symbol=ticker), **kw)
        )

    # Rating

    async def rating(
        self,
        ticker: Annotated[str, Doc("Ticker symbol")],
        **kw: Any,
    ) -> list[dict]:
        """Company rating - real time"""
        return await self.get_path(f"rating/{ticker}", **kw)

    async def etf_holders(
        self,
        ticker: Annotated[str, Doc("Ticker symbol")],
        **kw: Any,
    ) -> list[dict]:
        return await self.get_path(f"etf-holder/{ticker}", **kw)

    async def ratios(
        self,
        ticker: Annotated[str, Doc("Ticker symbol")],
        period: Annotated[
            str | None, Doc("Reporting period (e.g., 'annual', 'quarter')")
        ] = None,
        limit: Annotated[int | None, Doc("Maximum number of results")] = None,
        **kw: Any,
    ) -> list[dict]:
        """Company financial ratios - if period not provided it is for
        the trailing 12 months"""
        path = "ratios" if period else "ratios-ttm"
        return await self.get_path(
            f"{path}/{ticker}",
            **self.params(compact_dict(period=period, limit=limit), **kw),
        )

    async def peers(
        self,
        *tickers: Annotated[str, Doc("One or more ticker symbols")],
        **kw: Any,
    ) -> list[dict]:
        """Stock peers based on sector, exchange and market cap"""
        kwargs = self.params(**kw)
        kwargs["params"]["symbol"] = self.join(*tickers)
        return await self.get_path("stock_peers", **kwargs)

    async def news(
        self,
        *tickers: Annotated[str, Doc("One or more ticker symbols")],
        **kw: Any,
    ) -> list[dict]:
        """Company quote - real time"""
        kwargs = self.params(**kw)
        if tickers:
            kwargs["params"]["tickers"] = self.join(*tickers)
        return await self.get_path("stock_news", **kwargs)

    async def search(
        self,
        query: Annotated[str, Doc("Search query string")],
        *,
        exchange: Annotated[str | None, Doc("Filter by exchange")] = None,
        limit: Annotated[int | None, Doc("Maximum number of results")] = None,
        symbol: Annotated[bool, Doc("Search by symbol instead of name")] = False,
        **kw: Any,
    ) -> list[dict]:
        path = "search-symbol" if symbol else "search-name"
        return await self.get_path(
            path,
            **self.params(
                compact_dict(query=query, exchange=exchange, limit=limit), **kw
            ),
        )

    async def prices(
        self,
        symbol: Annotated[str, Doc("Ticker symbol")],
        *,
        frequency: Annotated[
            str | None, Doc("Price frequency (e.g., '1min', '5min', 'daily')")
        ] = None,
        from_date: Annotated[
            str | date | None, Doc("From date for historical prices")
        ] = None,
        to_date: Annotated[
            str | date | None, Doc("To date for historical prices")
        ] = None,
        convert_to_date: Annotated[
            bool, Doc("Convert date column to datetime type")
        ] = False,
    ) -> pd.DataFrame:
        """Historical prices, daily if frequency is not provided"""
        path = (
            "historical-price-eod/full"
            if not frequency
            else f"historical-chart/{frequency}"
        )
        data = await self.get_path(
            path,
            params=compact_dict(
                {"from": to_date_iso(from_date), "to": to_date_iso(to_date)},
                frequency=frequency,
                symbol=symbol,
            ),
        )
        if isinstance(data, dict):
            data = data.get("historical", [])
        df = pd.DataFrame(data)
        if convert_to_date and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # Sector performance
    async def sector_performance(
        self,
        *,
        from_date: Annotated[
            date | None, Doc("Start date for historical sector performance")
        ] = None,
        to_date: Annotated[
            date | None, Doc("End date for historical sector performance")
        ] = None,
        summary: Annotated[
            bool, Doc("Return summary instead of daily performance")
        ] = False,
        params: Annotated[dict | None, Doc("Additional query parameters")] = None,
        **kw: Any,
    ) -> dict | list[dict]:
        if not from_date:
            data = await self.get_path("sectors-performance", params=params, **kw)
            return {d["sector"]: Decimal(d["changesPercentage"][:-1]) for d in data}
        else:
            params = params.copy() if params is not None else {}
            params.update(compact_dict({"from": from_date, "to": to_date}))
            data = await self.get_path(
                "historical-sectors-performance", params=params, **kw
            )
            ts = [dict(nice_sector_performance(d)) for d in data]
            return summary_sector_performance(ts) if summary else ts

    async def sector_pe(self, **kw: Any) -> list[dict]:
        return cast(list[dict], await self.get_path("sector_price_earning_ratio", **kw))

    # forex
    async def forex_list(self) -> list[dict]:
        return await self.get_path("symbol/available-forex-currency-pairs")

    def historical_frequencies(self) -> dict:
        return {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "1hour": 60,
            "4hour": 240,
            "": 1440,
        }

    def historical_frequencies_annulaized(self) -> dict:
        one_year = 525600
        return {k: v / one_year for k, v in self.historical_frequencies().items()}

    # Crypto
    async def crypto_list(self) -> list[dict]:
        return await self.get_path("symbol/available-cryptocurrencies")

    # Internals
    async def get_path(
        self,
        path: Annotated[str, Doc("API endpoint path")],
        **kw: Any,
    ) -> list[dict]:
        result = await self.get(f"{self.url}/{path}", **self.params(**kw))
        return cast(list[dict], result)

    def join(
        self,
        *tickers: Annotated[str, Doc("One or more ticker symbols")],
    ) -> str:
        value = ",".join(tickers)
        if not value:
            raise TypeError("at least one ticker must be provided")
        return value

    def params(
        self,
        params: Annotated[dict | None, Doc("Query parameters dictionary")] = None,
        **kw: Any,
    ) -> dict:
        params = params.copy() if params is not None else {}
        params["apikey"] = self.key
        return {"params": params, **kw}


def nice_sector_performance(d: dict) -> Iterator[tuple[str, Any]]:
    for k, v in d.items():
        if k == "date":
            yield k, date.fromisoformat(v)
        else:
            kk = " ".join(w.title() for w in inflection.underscore(k).split("_")[:-2])
            yield kk, v


def summary_sector_performance(days: list[dict]) -> dict:
    result = days[0].copy()
    result.pop("date", None)
    for d in days[1:]:
        for k, v in d.items():
            if k != "date":
                p = result.get(k, 0)
                result[k] = 100 * ((1.0 + p / 100) * (1.0 + v / 100) - 1.0)
    return dict((k, to_decimal(round(v, 3))) for k, v in result.items())
