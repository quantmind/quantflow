from typing import Any

import pandas as pd
from fluid.utils.http_client import AioHttpClient, HttpResponse


class Deribit(AioHttpClient):
    url = "https://www.deribit.com/api/v2"

    async def get_book_summary_by_instrument(self, **kw: Any) -> Any:
        return await self.get_path("public/get_book_summary_by_instrument", **kw)

    async def get_volatility(self, **kw: Any) -> pd.DataFrame:
        kw.update(callback=self.to_df)
        return await self.get_path("public/get_historical_volatility", **kw)

    async def get_path(self, path: str, **kw: Any) -> dict:
        return await self.get(f"{self.url}/{path}", **kw)

    async def to_df(self, response: HttpResponse) -> pd.DataFrame:
        data = await response.json()
        df = pd.DataFrame(data["result"], columns=["timestamp", "volatility"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
