from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import StrEnum
from pathlib import Path

import pandas as pd
from fluid.utils.http_client import HttpResponse, HttpxClient, ResponseType
from typing_extensions import Annotated, Doc

from quantflow.options.inputs import DefaultVolSecurity, OptionType
from quantflow.options.surface import VolSurfaceLoader
from quantflow.rates.yield_curve import NoDiscount
from quantflow.utils.dates import as_utc, utcnow
from quantflow.utils.numbers import to_decimal


@dataclass
class Yahoo(HttpxClient):
    """Yahoo Finance API client

    Minimal client for fetching historical prices and option chains.

    ## Examples

    Fetch daily prices for a symbol:

    ```python
    from quantflow.data.yahoo import Yahoo

    async with Yahoo() as yahoo:
        df = await yahoo.prices("AAPL", range="1y")
    ```

    Build a volatility surface from the option chain:

    ```python
    async with Yahoo() as yahoo:
        loader = await yahoo.volatility_surface_loader("AAPL")
        surface = loader.surface()
    ```
    """

    url: str = "https://query2.finance.yahoo.com/v7/finance"
    content_type: str = (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    )
    default_headers: dict[str, str] = field(
        default_factory=lambda: {
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
            )
        }
    )
    _crumb: str | None = None

    class freq(StrEnum):
        """Yahoo Finance chart intervals"""

        one_min = "1m"
        two_min = "2m"
        five_min = "5m"
        fifteen_min = "15m"
        thirty_min = "30m"
        one_hour = "1h"
        one_day = "1d"
        five_day = "5d"
        one_week = "1wk"
        one_month = "1mo"
        three_month = "3mo"

    async def option_chain(
        self,
        symbol: Annotated[str, Doc("Underlying ticker symbol")],
    ) -> dict:  # pragma: no cover
        """Return the full option chain for `symbol`"""
        params = dict(getAllData="true", crumb=await self._get_crumb())
        data = await self.get(f"{self.url}/options/{symbol}", params=params)
        return data["optionChain"]["result"][0]

    async def volatility_surface_loader(
        self,
        symbol: Annotated[str, Doc("Underlying ticker symbol")],
        *,
        ref_date: Annotated[
            datetime | None,
            Doc("Reference date for the yield curves; defaults to now"),
        ] = None,
        exclude_volume: Annotated[
            int | None, Doc("Drop contracts with volume at or below this threshold")
        ] = None,
        exclude_open_interest: Annotated[
            int | None,
            Doc("Drop contracts with open interest at or below this threshold"),
        ] = None,
    ) -> VolSurfaceLoader:
        """Build a [VolSurfaceLoader][quantflow.options.surface.VolSurfaceLoader]
        by fetching the option chain for `symbol` and passing it to
        [loader_from_chain][quantflow.data.yahoo.Yahoo.loader_from_chain]."""
        return self.loader_from_chain(
            await self.option_chain(symbol),
            ref_date=ref_date,
            exclude_volume=exclude_volume,
            exclude_open_interest=exclude_open_interest,
        )

    @classmethod
    def loader_from_chain(
        cls,
        chain: Annotated[dict, Doc("Yahoo option chain payload")],
        *,
        ref_date: Annotated[
            datetime | None,
            Doc("Reference date for the yield curves; defaults to now"),
        ] = None,
        exclude_volume: Annotated[
            int | None, Doc("Drop contracts with volume at or below this threshold")
        ] = None,
        exclude_open_interest: Annotated[
            int | None,
            Doc("Drop contracts with open interest at or below this threshold"),
        ] = None,
    ) -> VolSurfaceLoader:
        """Build a [VolSurfaceLoader][quantflow.options.surface.VolSurfaceLoader]
        from a Yahoo option chain dictionary.

        US equity options are non-inverse: prices are in the quote currency and
        the spot is taken from the underlying quote. Forwards are not provided
        by Yahoo, so they are recovered from put-call parity by the loader.
        """
        symbol = chain.get("underlyingSymbol", "")
        ref = ref_date or utcnow()
        loader = VolSurfaceLoader(
            asset=symbol,
            exclude_volume=to_decimal(exclude_volume) if exclude_volume else None,
            exclude_open_interest=(
                to_decimal(exclude_open_interest) if exclude_open_interest else None
            ),
            quote_curve=NoDiscount(ref_date=ref),
            asset_curve=NoDiscount(ref_date=ref),
        )
        quote = chain.get("quote") or {}
        bid = quote.get("bid") or quote.get("regularMarketPrice")
        ask = quote.get("ask") or quote.get("regularMarketPrice")
        if bid and ask:
            loader.add_spot(
                DefaultVolSecurity.spot(),
                bid=to_decimal(bid),
                ask=to_decimal(ask),
            )
        for expiry in chain.get("options", []):
            maturity = (
                pd.to_datetime(expiry["expirationDate"], unit="s", utc=True)
                .to_pydatetime()
                .replace(hour=20, tzinfo=timezone.utc)
            )
            for option_type, contracts in (
                (OptionType.call, expiry.get("calls", [])),
                (OptionType.put, expiry.get("puts", [])),
            ):
                for c in contracts:
                    bid_ = c.get("bid")
                    ask_ = c.get("ask")
                    if not bid_ or not ask_:
                        continue
                    loader.add_option(
                        DefaultVolSecurity.option(),
                        strike=to_decimal(c["strike"]),
                        maturity=maturity,
                        option_type=option_type,
                        bid=to_decimal(bid_),
                        ask=to_decimal(ask_),
                        open_interest=to_decimal(c.get("openInterest") or 0),
                        volume=to_decimal(c.get("volume") or 0),
                        inverse=False,
                    )
        return loader

    async def prices(
        self,
        symbol: Annotated[str, Doc("Ticker symbol")],
        *,
        interval: Annotated[
            str | freq, Doc("Bar interval — use Yahoo.freq members or a raw string")
        ] = freq.one_day,
        from_date: Annotated[date | None, Doc("Start date (inclusive)")] = None,
        to_date: Annotated[date | None, Doc("End date (inclusive)")] = None,
        range: Annotated[
            str | None,
            Doc(
                "Shorthand period when dates are omitted: '1mo', '3mo', '6mo', "
                "'1y', '2y', '5y', 'ytd', 'max', etc."
            ),
        ] = None,
    ) -> pd.DataFrame:
        """Historical OHLCV prices for `symbol`.

        Returns a DataFrame with columns `timestamp`, `open`, `high`, `low`,
        `close`, `volume`, and `adj_close` (when available).

        Pass `from_date` / `to_date` for a specific window, or `range` for a
        shorthand period. When neither is given Yahoo defaults to one month.
        """
        params: dict = {"interval": str(interval)}
        if from_date:
            params["period1"] = int(as_utc(from_date).timestamp())
        if to_date:
            params["period2"] = int(as_utc(to_date).timestamp())
        if range and not from_date and not to_date:
            params["range"] = range
        data = await self.get(
            f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
            params=params,
        )
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamps") or result.get("timestamp", [])
        quote = result["indicators"]["quote"][0]
        adj = result["indicators"].get("adjclose", [{}])[0]
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            }
        )
        if "adjclose" in adj:
            df["adj_close"] = adj["adjclose"]
        return df

    async def save_fixture(
        self,
        symbol: Annotated[str, Doc("Underlying ticker symbol")],
        path: Annotated[str | Path, Doc("File path where to save the fixture")],
    ) -> Path:
        """Fetch the option chain for `symbol` and save it as a JSON fixture.

        Only the fields read by
        [volatility_surface_loader]
        [quantflow.data.yahoo.Yahoo.volatility_surface_loader]
        are kept, so the fixture stays small enough to commit.

        If `path` ends with `.gz`, the output is gzipped.
        """
        chain = await self.option_chain(symbol)
        contract_keys = ("strike", "bid", "ask", "openInterest", "volume")
        quote_keys = ("bid", "ask", "regularMarketPrice")
        quote = chain.get("quote") or {}
        stripped = {
            "underlyingSymbol": chain.get("underlyingSymbol", symbol),
            "quote": {k: quote[k] for k in quote_keys if k in quote},
            "options": [
                {
                    "expirationDate": expiry["expirationDate"],
                    "calls": [
                        {k: c[k] for k in contract_keys if k in c}
                        for c in expiry.get("calls", [])
                    ],
                    "puts": [
                        {k: c[k] for k in contract_keys if k in c}
                        for c in expiry.get("puts", [])
                    ],
                }
                for expiry in chain.get("options", [])
            ],
        }
        out = Path(path)
        payload = json.dumps(stripped, indent=2).encode()
        if out.suffix == ".gz":
            out.write_bytes(gzip.compress(payload))
        else:
            out.write_bytes(payload)
        return out

    async def _get_crumb(self) -> str:  # pragma: no cover
        if self._crumb is not None:
            return self._crumb
        text = await self.get("https://query2.finance.yahoo.com/v1/test/getcrumb")
        self._crumb = text.strip()
        return self._crumb

    @classmethod
    async def response_data(
        cls, response: HttpResponse
    ) -> ResponseType:  # pragma: no cover
        if (
            "text/plain" in response.headers["content-type"]
            or "text/html" in response.headers["content-type"]
        ):
            return await response.text()
        return await response.json()
