from datetime import datetime, timezone
from typing import Any, cast

import pandas as pd
from dateutil.parser import parse
from fluid.utils.http_client import AioHttpClient, HttpResponse

from quantflow.options.surface import VolSecurityType, VolSurfaceLoader
from quantflow.utils.numbers import round_to_step


def parse_maturity(v: str) -> datetime:
    return parse(v).replace(tzinfo=timezone.utc, hour=8)


class Deribit(AioHttpClient):
    url = "https://www.deribit.com/api/v2"

    async def get_book_summary_by_instrument(self, **kw: Any) -> list[dict]:
        kw.update(callback=self.to_result)
        return cast(
            list[dict],
            await self.get_path("public/get_book_summary_by_instrument", **kw),
        )

    async def get_book_summary_by_currency(self, **kw: Any) -> list[dict]:
        kw.update(callback=self.to_result)
        return cast(
            list[dict], await self.get_path("public/get_book_summary_by_currency", **kw)
        )

    async def get_instruments(self, **kw: Any) -> list[dict]:
        kw.update(callback=self.to_result)
        return cast(list[dict], await self.get_path("public/get_instruments", **kw))

    async def get_volatility(self, **kw: Any) -> pd.DataFrame:
        kw.update(callback=self.to_df)
        return await self.get_path("public/get_historical_volatility", **kw)

    async def volatility_surface_loader(self, currency: str) -> VolSurfaceLoader:
        """Create the volatility surface loader for a given crypto-currency"""
        loader = VolSurfaceLoader()
        futures = await self.get_book_summary_by_currency(
            params=dict(currency=currency, kind="future")
        )
        options = await self.get_book_summary_by_currency(
            params=dict(currency=currency, kind="option")
        )
        instruments = await self.get_instruments(params=dict(currency=currency))
        instrument_map = {i["instrument_name"]: i for i in instruments}

        for future in futures:
            if (bid_ := future["bid_price"]) and (ask_ := future["ask_price"]):
                name = future["instrument_name"]
                meta = instrument_map[name]
                tick_size = meta["tick_size"]
                bid = round_to_step(bid_, tick_size)
                ask = round_to_step(ask_, tick_size)
                if meta["settlement_period"] == "perpetual":
                    loader.add_spot(
                        VolSecurityType.spot,
                        bid=bid,
                        ask=ask,
                        open_interest=int(future["open_interest"]),
                        volume=int(future["volume_usd"]),
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
                        open_interest=int(future["open_interest"]),
                        volume=int(future["volume_usd"]),
                    )

        for option in options:
            if (bid_ := option["bid_price"]) and (ask_ := option["ask_price"]):
                name = option["instrument_name"]
                meta = instrument_map[name]
                tick_size = meta["tick_size"]
                loader.add_option(
                    VolSecurityType.option,
                    strike=round_to_step(meta["strike"], tick_size),
                    maturity=pd.to_datetime(
                        meta["expiration_timestamp"],
                        unit="ms",
                        utc=True,
                    ).to_pydatetime(),
                    call=meta["option_type"] == "call",
                    bid=round_to_step(bid_, tick_size),
                    ask=round_to_step(ask_, tick_size),
                )

        return loader

    # Internal methods

    async def get_path(self, path: str, **kw: Any) -> dict:
        return await self.get(f"{self.url}/{path}", **kw)

    async def to_result(self, response: HttpResponse) -> list[dict]:
        data = await response.json()
        return cast(list[dict], data["result"])

    async def to_df(self, response: HttpResponse) -> pd.DataFrame:
        data = await self.to_result(response)
        df = pd.DataFrame(data, columns=["timestamp", "volatility"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
