from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, cast

import pandas as pd
from dateutil.parser import parse
from fluid.utils.data import compact_dict
from fluid.utils.http_client import AioHttpClient, HttpResponse, HttpResponseError
from typing_extensions import Annotated, Doc

from quantflow.options.inputs import OptionType
from quantflow.options.surface import VolSecurityType, VolSurfaceLoader
from quantflow.utils.numbers import (
    Number,
    round_to_step,
    to_decimal,
    to_decimal_or_none,
)


def parse_maturity(v: str) -> datetime:
    return parse(v).replace(tzinfo=timezone.utc, hour=8)


class InstrumentKind(enum.StrEnum):
    """Instrument kind for Deribit API."""

    future = enum.auto()
    option = enum.auto()
    spot = enum.auto()
    future_combo = enum.auto()
    option_combo = enum.auto()


@dataclass
class Deribit(AioHttpClient):
    """Deribit API client

    ## Example

    ```python
    from quantflow.data.deribit import Deribit

    deribit = Deribit()
    ```
    """

    url: str = "https://www.deribit.com/api/v2"

    async def get_book_summary_by_instrument(
        self,
        instrument_name: Annotated[str, Doc("Instrument name")],
        **kw: Any,
    ) -> list[dict]:
        """Get the book summary for a given instrument."""
        kw.update(params=dict(instrument_name=instrument_name), callback=self.to_result)
        return cast(
            list[dict],
            await self.get_path("public/get_book_summary_by_instrument", **kw),
        )

    async def get_book_summary_by_currency(
        self,
        currency: Annotated[str, Doc("Currency")],
        kind: Annotated[InstrumentKind | None, Doc("Optional instrument kind")] = None,
        **kw: Any,
    ) -> list[dict]:
        """Get the book summary for a given currency."""
        kw.update(
            params=compact_dict(currency=currency, kind=kind), callback=self.to_result
        )
        return cast(
            list[dict], await self.get_path("public/get_book_summary_by_currency", **kw)
        )

    async def get_instruments(
        self,
        currency: Annotated[str, Doc("Currency")],
        kind: Annotated[InstrumentKind | None, Doc("Optional instrument kind")] = None,
        expired: Annotated[bool | None, Doc("Include expired instruments")] = None,
        **kw: Any,
    ) -> list[dict]:
        """Get the list of instruments for a given currency."""
        kw.update(
            params=compact_dict(currency=currency, kind=kind, expired=expired),
            callback=self.to_result,
        )
        return cast(list[dict], await self.get_path("public/get_instruments", **kw))

    async def get_volatility(
        self,
        currency: Annotated[str, Doc("Currency")],
        **kw: Any,
    ) -> pd.DataFrame:
        """Provides information about historical volatility for given cryptocurrency"""
        kw.update(params=dict(currency=currency), callback=self.to_df)
        return await self.get_path("public/get_historical_volatility", **kw)

    async def volatility_surface_loader(
        self,
        currency: Annotated[str, Doc("Currency")],
        *,
        exclude_open_interest: Annotated[
            Number | None,
            Doc("Exclude options with open interest below this threshold"),
        ] = None,
        exclude_volume: Annotated[
            Number | None, Doc("Exclude options with volume below this threshold")
        ] = None,
    ) -> VolSurfaceLoader:
        """Create a :class:`.VolSurfaceLoader` for a given crypto-currency"""
        loader = VolSurfaceLoader(
            asset=currency,
            exclude_open_interest=to_decimal_or_none(exclude_open_interest),
            exclude_volume=to_decimal_or_none(exclude_volume),
        )
        futures = await self.get_book_summary_by_currency(
            currency=currency, kind=InstrumentKind.future
        )
        options = await self.get_book_summary_by_currency(
            currency=currency, kind=InstrumentKind.option
        )
        instruments = await self.get_instruments(currency=currency)
        instrument_map = {i["instrument_name"]: i for i in instruments}
        min_tick_size = Decimal("inf")
        for entry in futures:
            if (bid_ := entry["bid_price"]) and (ask_ := entry["ask_price"]):
                name = entry["instrument_name"]
                meta = instrument_map[name]
                tick_size = to_decimal(meta["tick_size"])
                min_tick_size = min(min_tick_size, tick_size)
                bid = round_to_step(bid_, tick_size)
                ask = round_to_step(ask_, tick_size)
                if meta["settlement_period"] == "perpetual":
                    loader.add_spot(
                        VolSecurityType.spot,
                        bid=bid,
                        ask=ask,
                        open_interest=to_decimal(entry["open_interest"]),
                        volume=to_decimal(entry["volume_usd"]),
                    )
                else:
                    maturity = pd.to_datetime(
                        meta["expiration_timestamp"],
                        unit="ms",
                        utc=True,
                    ).to_pydatetime()
                    loader.add_forward(
                        VolSecurityType.forward,
                        maturity=maturity,
                        bid=bid,
                        ask=ask,
                        open_interest=to_decimal(entry["open_interest"]),
                        volume=to_decimal(entry["volume_usd"]),
                    )
        loader.tick_size_forwards = min_tick_size

        min_tick_size = Decimal("inf")
        for entry in options:
            if (bid_ := entry["bid_price"]) and (ask_ := entry["ask_price"]):
                name = entry["instrument_name"]
                meta = instrument_map[name]
                tick_size = to_decimal(meta["tick_size"])
                min_tick_size = min(min_tick_size, tick_size)
                loader.add_option(
                    VolSecurityType.option,
                    strike=round_to_step(meta["strike"], tick_size),
                    maturity=pd.to_datetime(
                        meta["expiration_timestamp"],
                        unit="ms",
                        utc=True,
                    ).to_pydatetime(),
                    option_type=(
                        OptionType.call
                        if meta["option_type"] == "call"
                        else OptionType.put
                    ),
                    bid=round_to_step(bid_, tick_size),
                    ask=round_to_step(ask_, tick_size),
                    open_interest=to_decimal(entry["open_interest"]),
                    volume=to_decimal(entry["volume_usd"]),
                )
        loader.tick_size_options = min_tick_size
        return loader

    # Internal methods

    async def get_path(
        self,
        path: Annotated[str, Doc("API path")],
        **kw: Any,
    ) -> dict:
        return await self.get(f"{self.url}/{path}", **kw)

    async def to_result(
        self,
        response: Annotated[HttpResponse, Doc("HTTP response object")],
    ) -> list[dict]:
        data = await response.json()
        if "error" in data:
            raise HttpResponseError(response, data["error"])
        return cast(list[dict], data["result"])

    async def to_df(
        self,
        response: Annotated[HttpResponse, Doc("HTTP response object")],
    ) -> pd.DataFrame:
        data = await self.to_result(response)
        df = pd.DataFrame(data, columns=["timestamp", "volatility"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
