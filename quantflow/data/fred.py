from .client import AioHttpClient
from dataclasses import dataclass, field
from typing import Any, cast
import os


@dataclass
class Fred(AioHttpClient):
    url: str = "https://api.stlouisfed.org/fred"
    key: str = field(default_factory=lambda: os.environ.get("FRED_API_KEY", ""))

    async def categiories(self, **kw: Any) -> dict:
        return await self.get_path("category", **kw)

    async def subcategories(self, **kw: Any) -> dict:
        return await self.get_path("category/children", **kw)

    async def series(self, **kw: Any) -> dict:
        return await self.get_path("category/series", **kw)

    # Internals
    async def get_path(self, path: str, **kw: Any) -> dict:
        result = await self.get(f"{self.url}/{path}", **self.params(**kw))
        return cast(dict, result)

    def params(self, params: dict | None = None, **kw: Any) -> dict:
        params = params.copy() if params is not None else {}
        params.update(api_key=self.key, file_type="json")
        return {"params": params, **kw}
