import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.docs import load_description
from quantflow.dists.distributions1d import DoubleExponential
from quantflow.options.pricer import OptionPricer
from quantflow.sp.heston import HestonJ
from quantflow.sp.jump_diffusion import JumpDiffusion

heston_router = APIRouter()


class VolSurfaceGridResponse(BaseModel):
    moneyness: list[float] = Field(description="Moneyness grid values")
    ttm: list[float] = Field(description="Time to maturity grid values")
    iv: list[list[float]] = Field(
        description="Implied vol grid (rows=ttm, cols=moneyness)"
    )


@heston_router.get(
    "/heston-vol-surface",
    summary="Theoretical implied volatility surface",
    description=load_description("heston_vol_surface.md"),
)
async def heston_vol_surface(
    model: str = Query(
        "jd",
        description="Model type",
        enum=["jd", "hj"],
    ),
    vol: float = Query(0.4, description="Long term volatility", ge=0.1, le=0.8),
    sigma: float = Query(0.5, description="Vol of vol", ge=0.1, le=2.0),
    kappa: float = Query(0.5, description="Variance mean reversion", ge=0.1, le=2.0),
    rho: float = Query(0.0, description="Correlation", ge=-0.6, le=0.6),
    r: float = Query(1.0, description="Initial vol ratio", ge=0.6, le=1.6),
    jump_fraction: float = Query(0.5, description="Jump fraction", ge=0.1, le=0.9),
    jump_intensity: float = Query(
        10.0, description="Jump intensity", ge=10.0, le=100.0
    ),
    jump_asymmetry: float = Query(
        0.0, description="Jump asymmetry (log kappa)", ge=-2.0, le=2.0
    ),
) -> VolSurfaceGridResponse:
    pricer: OptionPricer
    if model == "jd":
        vm_jd = JumpDiffusion.create(
            DoubleExponential,
            vol=vol,
            jump_fraction=jump_fraction,
            jump_intensity=jump_intensity,
            jump_asymmetry=jump_asymmetry,
        )
        pricer = OptionPricer(model=vm_jd)
    else:
        st = sigma / vol
        k = max(kappa, 0.5 * st * st)
        vm_hj = HestonJ.create(
            DoubleExponential,
            rate=r,
            vol=vol,
            sigma=sigma,
            kappa=k,
            rho=rho,
            jump_fraction=jump_fraction,
            jump_intensity=jump_intensity,
            jump_asymmetry=jump_asymmetry,
        )
        pricer = OptionPricer(model=vm_hj)
    ttm_arr = np.linspace(0.1, 1.0, 10)
    moneyness_arr = np.linspace(-0.5, 0.5, 51)
    implied = np.zeros((len(ttm_arr), len(moneyness_arr)))
    for i, t in enumerate(ttm_arr):
        maturity = pricer.maturity(float(t))
        vols = maturity.prices(moneyness_arr * np.sqrt(t))["iv"].values
        # replace NaN/Inf/negative with 0
        vols = np.where(np.isfinite(vols) & (vols > 0), vols, 0.0)
        implied[i, :] = vols

    return VolSurfaceGridResponse(
        moneyness=[float(m) for m in moneyness_arr],
        ttm=[float(t) for t in ttm_arr],
        iv=[[float(v) for v in row] for row in implied],
    )
