import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from fluid.utils.http_client import AioHttpClient

MATURITIES = (
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
)


@dataclass
class FederalReserve(AioHttpClient):
    """Federal Reserve API client.

    This class is used to fetch yield curves from the Federal Reserve at
    https://www.federalreserve.gov/datadownload/
    """

    url: str = "https://www.federalreserve.gov/datadownload/Output.aspx"
    default_params: dict[str, Any] = field(
        default_factory=lambda: {
            "from": "",
            "to": "",
            "lastobs": "",
            "filetype": "csv",
            "label": "include",
            "layout": "seriescolumn",
            "type": "package",
        }
    )

    async def yield_curves(self, **params: Any) -> pd.DataFrame:
        """Get treasury constant maturities rates"""
        params.update(series="bf17364827e38702b42a58cf8eaa3f78", rel="H15")
        data = await self._get_text(params)
        df = pd.read_csv(data, header=5, index_col=None, parse_dates=True)
        df.columns = list(("date",) + MATURITIES)  # type: ignore
        df = df.set_index("date").replace("ND", np.nan)
        return df.dropna(axis=0, how="all").reset_index()

    async def ref_rates(self, **params: Any) -> pd.DataFrame:
        """Get policy rates

        Prior to 2021-07-08 it is the rate on excess reserves (IOER rate)
        After 2021-07-08 it is the rate on reserve balances (IORB rate)

        The IOER rate was the primary tool used by the Federal Reserve to set
        a floor on the federal funds rate.
        While the Interest rate on required reserves (IORR rate) existed,
        the IOER rate had a more direct impact on market rates,
        as banks typically held far more excess reserves than required reserves.
        Therefore, the IOER rate was more influential
        in the Fed's monetary policy implementation.
        """
        params.update(series="c27939ee810cb2e929a920a6bd77d9f6", rel="PRATES")
        data = await self._get_text(params)
        df = pd.read_csv(data, header=5, index_col=None, parse_dates=True)
        ioer = df["RESBME_N.D"]
        iorb = df["RESBM_N.D"]
        rate = iorb.combine_first(ioer)
        return pd.DataFrame(
            {
                "date": df["Time Period"],
                "rate": rate,
            }
        )

    async def _get_text(self, params: dict[str, Any]) -> io.StringIO:
        """Get parameters for the request."""
        params = {**self.default_params, **params}
        response = await self.get(self.url, params=params, callback=True)
        data = await response.text()
        return io.StringIO(data)
