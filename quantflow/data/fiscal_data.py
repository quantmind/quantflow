from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from fluid.utils.http_client import AioHttpClient

from quantflow.utils.dates import as_date

URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "rel=H15&series=bf17364827e38702b42a58cf8eaa3f78&lastobs=&"
)

maturities = [
    "month_1",
    "month_3",
    "month_6",
    "year_1",
    "year_2",
    "year_3",
    "year_5",
    "year_7",
    "year_10",
    "year_20",
    "year_30",
]


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
        url = f"{self.url}{path}"
        payload = await self.get(url, params=params)
        data = payload["data"]
        return data
