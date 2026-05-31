import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from fluid.utils.http_client import AioHttpClient
from typing_extensions import Annotated, Doc

MATURITIES = (
    "1M",
    "3M",
    "6M",
    "1Y",
    "2Y",
    "3Y",
    "5Y",
    "7Y",
    "10Y",
    "20Y",
    "30Y",
)


def get_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "from": "",
        "to": "",
        "lastobs": "",
        "filetype": "csv",
        "label": "include",
        "layout": "seriescolumn",
        "type": "package",
        **params,
    }


@dataclass
class FederalReserve(AioHttpClient):
    """Federal Reserve API client.

    This class is used to fetch yield curves from the Federal Reserve
    [data download](https://www.federalreserve.gov/datadownload/)
    """

    url: Annotated[
        str,
        Doc("Base URL for the Federal Reserve data download API."),
    ] = "https://www.federalreserve.gov/datadownload/Output.aspx"

    async def yield_curves(self, **params: Any) -> pd.DataFrame:
        """Treasury constant maturity par yields indexed by observation date.

        Columns are the tenors in ``MATURITIES`` and rates are returned as
        decimals (for example ``0.0372`` for 3.72%).
        """
        params.update(series="bf17364827e38702b42a58cf8eaa3f78", rel="H15")
        data = await self._get_text(params)
        df = pd.read_csv(data, header=5, index_col=None, parse_dates=True)
        df.columns = list(("date",) + MATURITIES)  # type: ignore
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").replace("ND", np.nan)
        df = df.apply(pd.to_numeric, errors="coerce") / 100.0
        return df.dropna(axis=0, how="all")

    async def ref_rates(self, **params: Any) -> pd.DataFrame:
        """Get a pandas dataframe of policy rates

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
        params = get_params(params)
        response = await self.get(self.url, params=params, callback=True)
        data = await response.text()
        return io.StringIO(data)
