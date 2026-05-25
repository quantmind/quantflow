from datetime import date, timedelta
from functools import partial

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.docs import load_description
from quantflow.ta.ewma import EWMA
from quantflow.ta.supersmoother import SuperSmoother

from .deps import FMPDep, RedisDataframe, RedisDep

smoother_router = APIRouter()


class SmootherPoint(BaseModel):
    date: str = Field(description="Date string")
    close: float = Field(description="Close price")
    supersmoother: float = Field(description="SuperSmoother filtered value")
    ewma: float = Field(description="EWMA filtered value")


class SmootherResponse(BaseModel):
    data: list[SmootherPoint] = Field(description="Time series with smoothed values")


@smoother_router.get(
    "/supersmoother",
    summary="SuperSmoother and EWMA price filter",
    description=load_description("supersmoother.md"),
    response_model=SmootherResponse,
)
async def supersmoother(
    redis: RedisDep,
    fmp: FMPDep,
    period: int = Query(10, description="Filter period", ge=2, le=100),
    symbol: str = Query("BTCUSD", description="Ticker symbol"),
) -> SmootherResponse:
    cache = RedisDataframe(
        redis=redis,
        key=f"smoother:{symbol}",
    )
    start = date.today() - timedelta(days=365 * 2)
    prices = await cache.from_cache(partial(fmp.prices, symbol, from_date=start))
    sm = (
        prices[["date", "close"]]
        .copy()
        .sort_values("date", ascending=True)
        .reset_index(drop=True)
    )
    sm["supersmoother"] = sm["close"].apply(SuperSmoother(period=period).update)
    sm["ewma"] = sm["close"].apply(EWMA(period=period).update)
    sm["date"] = sm["date"].astype(str)
    return SmootherResponse(data=sm.to_dict(orient="records"))
