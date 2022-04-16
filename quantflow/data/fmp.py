import os
from typing import Dict, List, Optional

import pandas as pd

from .client import HttpClient, compact


class FMP(HttpClient):
    url = "https://financialmodelingprep.com/api"

    def __init__(
        self,
        key: str = "",
    ) -> None:
        self.key = key or os.environ.get("FMP_API_KEY")

    async def stocks(self, **kw) -> List[Dict]:
        return await self.get_path("v3/stock/list", **kw)

    async def etfs(self, **kw) -> List[Dict]:
        return await self.get_path("v3/etf/list", **kw)

    async def indices(self, **kw) -> List[Dict]:
        return await self.get_path("v3/quotes/index", **kw)

    async def profile(self, *tickers: str, **kw) -> List[Dict]:
        """Company profile - minute"""
        return await self.get_path(f"v3/profile/{self.join(*tickers)}", **kw)

    async def quote(self, *tickers: str, **kw) -> List[Dict]:
        """Company quote - real time"""
        return await self.get_path(f"v3/quote/{self.join(*tickers)}", **kw)

    async def executives(self, ticker: str, **kw) -> List[Dict]:
        """Company quote - real time"""
        return await self.get_path(f"v3/key-executives/{ticker}", **kw)

    async def rating(self, ticker: str, **kw) -> List[Dict]:
        """Company quote - real time"""
        return await self.get_path(f"v3/rating/{ticker}", **kw)

    async def etf_holders(self, ticker: str, **kw) -> List[Dict]:
        return await self.get_path(f"v3/etf-holder/{ticker}", **kw)

    async def ratios(
        self,
        ticker: str,
        period: Optional[str] = None,
        limit: Optional[int] = None,
        **kw,
    ) -> List[Dict]:
        """Company financial ratios - if period not provided it is for
        the trailing 12 months"""
        path = "ratios" if period else "ratios-ttm"
        return await self.get_path(
            f"v3/{path}/{ticker}",
            **self.params(compact(period=period, limit=limit), **kw),
        )

    async def peers(self, *tickers: str, **kw) -> List[Dict]:
        """Stock peers based on sector, exchange and market cap"""
        kwargs = self.params(**kw)
        kwargs["params"]["symbol"] = self.join(*tickers)
        return await self.get_path("v4/stock_peers", **kwargs)

    async def news(self, *tickers: str, **kw) -> List[Dict]:
        """Company quote - real time"""
        kwargs = self.params(**kw)
        if tickers:
            kwargs["params"]["tickers"] = self.join(*tickers)
        return await self.get_path("v3/stock_news", **kwargs)

    async def search(
        self,
        query: str,
        *,
        exchange: Optional[str] = None,
        limit: Optional[int] = None,
        ticker: bool = False,
        **kw,
    ) -> List[Dict]:
        path = "v3/search-ticker" if ticker else "v3/search"
        return await self.get_path(
            path,
            **self.params(compact(query=query, exchange=exchange, limit=limit), **kw),
        )

    async def prices(self, ticker: str, frequency: str = "", **kw) -> pd.DataFrame:
        base = (
            "historical-price-full/"
            if not frequency
            else f"historical-chart/{frequency}"
        )
        data = await self.get_path(f"v3/{base}/{ticker}", **kw)
        if isinstance(data, dict):
            data = data.get("historical", [])
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

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

    # Internals
    def get_path(self, path: str, **kw) -> List[Dict]:
        return self.get(f"{self.url}/{path}", **self.params(**kw))

    def join(self, *tickers: str) -> str:
        value = ",".join(tickers)
        if not value:
            raise TypeError("at least one ticker must be provided")
        return value

    def params(self, params: Optional[Dict] = None, **kwargs) -> Dict:
        params = params.copy() if params is not None else {}
        params["apikey"] = self.key
        return {"params": params, **kwargs}
