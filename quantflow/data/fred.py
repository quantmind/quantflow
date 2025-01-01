import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import pandas as pd
from fluid.utils.http_client import AioHttpClient


@dataclass
class Fred(AioHttpClient):
    """Federal Reserve Economic Data API client

    Fetch economic data from `FRED`_.

    .. _FRED: https://fred.stlouisfed.org/
    """

    url: str = "https://api.stlouisfed.org/fred"
    key: str = field(default_factory=lambda: os.environ.get("FRED_API_KEY", ""))

    class freq(StrEnum):
        """Fred historical frequencies"""

        d = "d"
        w = "w"
        bw = "bw"
        m = "m"
        q = "q"
        sa = "sa"
        a = "a"

    async def categiories(self, **kw: Any) -> dict:
        """Get categories"""
        return await self.get_path("category", **kw)

    async def subcategories(self, **kw: Any) -> dict:
        """Get subcategories of a given category"""
        return await self.get_path("category/children", **kw)

    async def series(self, **kw: Any) -> dict:
        """Get series of a given category"""
        return await self.get_path("category/series", **kw)

    async def serie_data(self, *, to_date: bool = False, **kw: Any) -> pd.DataFrame:
        """Get series data frame"""
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
