from functools import partial

import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from quantflow.data.deribit import Deribit
from quantflow.options.inputs import VolSurfaceInputs
from quantflow.options.surface import OptionInfo
from quantflow.rates.nelson_siegel import NelsonSiegel

from .deps import RedisCache, RedisDep
from .rates import YieldCurveResponse

volatility_router = APIRouter()


class VolSurfaceResponse(BaseModel):
    inputs: VolSurfaceInputs = Field(description="Volatility surface inputs")
    options: list[OptionInfo] = Field(
        description="List of option info with implied volatilities"
    )
    quote_curve: YieldCurveResponse = Field(
        description="Quote discount curve with rates"
    )
    asset_curve: YieldCurveResponse = Field(
        description="Asset discount curve with rates"
    )


@volatility_router.get("/volatility-surface")
async def volatility_surface(
    redis: RedisDep,
    asset: str = Query(
        "btc",
        description="Crypto asset",
        enum=["btc", "eth"],
    ),
) -> VolSurfaceResponse:
    cache = RedisCache(
        redis=redis,
        Model=VolSurfaceResponse,
        key=f"vol_surface:{asset}",
    )
    return await cache.from_cache(partial(_volatility_surface, asset))


def _curve_response(curve, max_ttm: float) -> YieldCurveResponse:
    ttm = list(np.linspace(1 / 365, max_ttm, 50))
    rates = list(curve.continuously_compounded_rate(ttm))
    return YieldCurveResponse(curve=curve, ttm=ttm, rates=rates)


async def _volatility_surface(asset: str) -> VolSurfaceResponse:
    inverse = asset != "sol"
    async with Deribit() as cli:
        loader = await cli.volatility_surface_loader(asset, inverse=inverse)

    loader.calibrate_curves(quote_curve=NelsonSiegel)
    surface = loader.surface()
    surface.bs()
    surface.disable_outliers()

    inputs = surface.inputs(converged=True)
    options = [op.info() for op in surface.option_prices(converged=True)]

    max_ttm = max(float(op.ttm) for op in options) if options else 1.0

    return VolSurfaceResponse(
        inputs=inputs,
        options=options,
        quote_curve=_curve_response(surface.quote_curve, max_ttm),
        asset_curve=_curve_response(surface.asset_curve, max_ttm),
    )
