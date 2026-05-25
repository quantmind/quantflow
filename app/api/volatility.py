from functools import partial
from typing import Any

import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.docs import load_description
from quantflow.data.deribit import Deribit
from quantflow.data.yahoo import Yahoo
from quantflow.options.inputs import VolSurfaceInputs
from quantflow.options.surface import OptionInfo, VolSurfaceLoader
from quantflow.rates.cir import CIRCurve
from quantflow.rates.nelson_siegel import NelsonSiegel

from .deps import RedisCache, RedisDep
from .rates import YieldCurveResponse

volatility_router = APIRouter()

DERIBIT_ASSETS = {"BTC", "ETH"}
YAHOO_ASSETS = {"SPY", "AAPL", "NVDA"}
ALL_ASSETS = sorted(DERIBIT_ASSETS) + sorted(YAHOO_ASSETS)


class ForwardPoint(BaseModel):
    maturity: str = Field(description="Maturity date")
    ttm: float = Field(description="Time to maturity in years")
    forward: float = Field(description="Implied forward price from put-call parity")


class ForwardCurveResponse(BaseModel):
    ttm: list[float] = Field(description="Time to maturity in years")
    forward: list[float] = Field(
        description="Forward price from calibrated discount factors"
    )


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
    forward_curve: ForwardCurveResponse = Field(
        description="Model forward curve from calibrated discount factors"
    )
    pcp_forwards: list[ForwardPoint] = Field(
        description="Per-maturity implied forward from put-call parity"
    )


@volatility_router.get(
    "/volatility-surface",
    summary="Live implied volatility surface",
    description=load_description("volatility_surface.md"),
)
async def volatility_surface(
    redis: RedisDep,
    asset: str = Query(
        "BTC",
        description="Asset symbol",
        enum=ALL_ASSETS,
    ),
) -> VolSurfaceResponse:
    cache = RedisCache(
        redis=redis,
        Model=VolSurfaceResponse,
        key=f"vol_surface:{asset}",
    )
    return await cache.from_cache(partial(_volatility_surface, asset))


def _curve_response(curve: Any, max_ttm: float) -> YieldCurveResponse:
    ttm = list(np.linspace(1 / 365, max_ttm, 50))
    rates = [float(r) for r in np.atleast_1d(curve.continuously_compounded_rate(ttm))]
    return YieldCurveResponse(curve=curve, ttm=ttm, rates=rates)  # type: ignore[arg-type]


async def _load_surface(asset: str) -> VolSurfaceLoader:
    if asset in DERIBIT_ASSETS:
        async with Deribit() as cli:
            loader = await cli.volatility_surface_loader(asset.lower(), inverse=True)
        loader.calibrate_curves(quote_curve=CIRCurve, asset_curve=NelsonSiegel)
        return loader
    else:
        async with Yahoo() as cli:
            loader = await cli.volatility_surface_loader(asset)
        loader.calibrate_spot()
        loader.calibrate_curves(quote_curve=CIRCurve, asset_curve=NelsonSiegel)
        return loader


def _forward_curve_response(
    surface: Any, ttm_grid: list[float]
) -> ForwardCurveResponse:
    spot = float(surface.spot_price())
    forward = [
        spot
        * float(surface.asset_curve.discount_factor(t))
        / float(surface.quote_curve.discount_factor(t))
        for t in ttm_grid
    ]
    return ForwardCurveResponse(ttm=ttm_grid, forward=forward)


async def _volatility_surface(asset: str) -> VolSurfaceResponse:
    loader = await _load_surface(asset)
    surface = loader.surface()
    surface.bs()
    surface.disable_outliers()

    inputs = surface.inputs(converged=True)
    options = [op.info() for op in surface.option_prices(converged=True)]

    max_ttm = max(float(op.ttm) for op in options) if options else 1.0
    ttm_grid = list(np.linspace(1 / 365, max_ttm, 50))

    return VolSurfaceResponse(
        inputs=inputs,
        options=options,
        quote_curve=_curve_response(surface.quote_curve, max_ttm),
        asset_curve=_curve_response(surface.asset_curve, max_ttm),
        forward_curve=_forward_curve_response(surface, ttm_grid),
        pcp_forwards=[
            ForwardPoint(
                maturity=str(maturity)[:10],
                ttm=ttm,
                forward=forward,
            )
            for maturity, ttm, forward in loader.implied_forward_term_structure()
        ],
    )
