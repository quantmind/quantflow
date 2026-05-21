from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from quantflow.data.fmp import FMP

cointegration_router = APIRouter()


class CointegrationResponse(BaseModel):
    dates: list[str] = Field(description="Date strings")
    residuals: list[float] = Field(description="Cointegration residual values")
    deltas: list[float] = Field(
        description="Cointegrating vector (eigenvector for largest eigenvalue)"
    )


@cointegration_router.get("/cointegration")
async def cointegration(
    frequency: str = Query(
        "daily",
        description="Price frequency",
        enum=["1min", "5min", "15min", "30min", "1hour", "4hour", "daily"],
    ),
) -> CointegrationResponse:
    import numpy as np
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    # daily uses the EOD endpoint (frequency=None)
    freq = None if frequency == "daily" else frequency

    async with FMP() as cli:
        btc = await cli.prices("BTCUSD", convert_to_date=True, frequency=freq)
        eth = await cli.prices("ETHUSD", convert_to_date=True, frequency=freq)
        sol = await cli.prices("SOLUSD", convert_to_date=True, frequency=freq)

    btc = btc.set_index("date")
    eth = eth.set_index("date")
    sol = sol.set_index("date")

    prices_3 = (
        btc[["close"]]
        .join(eth[["close"]], lsuffix="_btc", rsuffix="_eth")
        .join(sol[["close"]])
    )
    prices_3.columns = ["btc_close", "eth_close", "sol_close"]
    prices_3 = prices_3.dropna()

    log_prices_3 = np.log(prices_3)
    johansen_result = coint_johansen(log_prices_3, det_order=0, k_ar_diff=1)
    deltas = johansen_result.evec[:, 0]

    residuals = log_prices_3.dot(deltas)
    residual_mean = residuals.mean()
    residuals = residuals - residual_mean

    dates = [str(d)[:10] for d in residuals.index]
    return CointegrationResponse(
        dates=dates,
        residuals=[float(v) for v in residuals.values],
        deltas=[float(v) for v in deltas],
    )
