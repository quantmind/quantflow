from functools import partial

import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from quantflow.data.fmp import FMP

from .deps import FMPDep, RedisCache, RedisDep

cointegration_router = APIRouter()


class CointegrationResponse(BaseModel):
    dates: list[str] = Field(description="Date strings")
    residuals: list[float] = Field(description="Cointegration residual values")
    deltas: list[float] = Field(
        description="Cointegrating vector (eigenvector for largest eigenvalue)"
    )


@cointegration_router.get("/cointegration")
async def cointegration(
    fmp: FMPDep,
    redis: RedisDep,
    frequency: FMP.freq = Query(
        FMP.freq.daily,
        description="Price frequency",
    ),
) -> CointegrationResponse:
    cache = RedisCache(
        redis=redis,
        Model=CointegrationResponse,
        key=f"cointegration:{frequency}",
    )
    return await cache.from_cache(partial(_cointegration, fmp, frequency))


async def _cointegration(fmp: FMP, frequency: FMP.freq) -> CointegrationResponse:
    btc = await fmp.prices("BTCUSD", convert_to_date=True, frequency=frequency)
    eth = await fmp.prices("ETHUSD", convert_to_date=True, frequency=frequency)
    sol = await fmp.prices("SOLUSD", convert_to_date=True, frequency=frequency)

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
