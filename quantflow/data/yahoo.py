from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path

import pandas as pd
from fluid.utils.http_client import HttpResponse, HttpxClient, ResponseType
from typing_extensions import Annotated, Doc

from quantflow.options.inputs import DefaultVolSecurity, OptionType
from quantflow.options.surface import VolSurfaceLoader
from quantflow.utils.numbers import to_decimal


@dataclass
class Yahoo(HttpxClient):
    """Yahoo Finance API client

    Minimal client for fetching option chains used to build volatility surfaces.

    ## Example

    ```python
    from quantflow.data.yahoo import Yahoo

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
            exclude_volume=exclude_volume,
            exclude_open_interest=exclude_open_interest,
        )

    @classmethod
    def loader_from_chain(
        cls,
        chain: Annotated[dict, Doc("Yahoo option chain payload")],
        *,
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
        loader = VolSurfaceLoader(
            asset=symbol,
            exclude_volume=to_decimal(exclude_volume) if exclude_volume else None,
            exclude_open_interest=(
                to_decimal(exclude_open_interest) if exclude_open_interest else None
            ),
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
