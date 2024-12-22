from .client import AioHttpClient
import pandas as pd
from enum import StrEnum
from dataclasses import dataclass, field
from typing import Any, cast
import os


@dataclass
class Fred(AioHttpClient):
    url: str = "https://api.stlouisfed.org/fred"
    key: str = field(default_factory=lambda: os.environ.get("FRED_API_KEY", ""))

    class freq(StrEnum):
        """FMP historical frequencies"""

        one_min = "1min"
        five_min = "5min"
        fifteen_min = "15min"
        thirty_min = "30min"
        one_hour = "1hour"
        four_hour = "4hour"
        daily = ""

    async def categiories(self, **kw: Any) -> dict:
        return await self.get_path("category", **kw)

    async def subcategories(self, **kw: Any) -> dict:
        return await self.get_path("category/children", **kw)

    async def series(self, **kw: Any) -> dict:
        return await self.get_path("category/series", **kw)

    async def serie_data(self, *, to_date: bool = False, **kw: Any) -> pd.DataFrame:
        data = await self.get_path("series/observations", **kw)
        df = pd.DataFrame(data["observations"])
        df["value"] = pd.to_numeric(df["value"])
        if to_date and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # Internals
    async def get_path(self, path: str, **kw: Any) -> dict:
        result = await self.get(f"{self.url}/{path}", **self.params(**kw))
        return cast(dict, result)

    def params(self, params: dict | None = None, **kw: Any) -> dict:
        params = params.copy() if params is not None else {}
        params.update(api_key=self.key, file_type="json")
        return {"params": params, **kw}
