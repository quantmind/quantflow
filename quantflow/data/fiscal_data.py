from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from fluid.utils.http_client import AioHttpClient

from quantflow.utils.dates import as_date


@dataclass
class FiscalData(AioHttpClient):
    """Fiscal Data API client.

    THis class is used to fetch data from the
    [fiscal data api](https://fiscaldata.treasury.gov/api-documentation/)
    """

    url: str = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

    async def securities(self, record_date: date | None = None) -> pd.DataFrame:
        """Get treasury constant maturities rates"""
        rd = as_date(record_date)
        pm = rd.replace(day=1) - timedelta(days=1)
        params = {"filter": f"record_date:eq:{pm.isoformat()}"}
        data = await self.get_all("/v1/debt/mspd/mspd_table_3_market", params)
        return pd.DataFrame(data)

    async def get_all(self, path: str, params: dict[str, str]) -> list:
        """Get all data from the API"""
        next_url: str | None = f"{self.url}{path}"
        full_data = []
        while next_url:
            payload = await self.get(next_url, params=params)
            full_data.extend(payload["data"])
            if links := payload.get("links"):
                if next_path := links.get("next"):
                    next_url = f"{self.url}{next_path}"
                else:
                    next_url = None
            else:
                next_url = None
        return full_data
