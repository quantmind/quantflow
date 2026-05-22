import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from quantflow.sp.ou import Vasicek
from quantflow.sp.poisson import PoissonProcess
from quantflow.utils import bins
from quantflow.utils.distributions import DoubleExponential

sampling_router = APIRouter()


class SamplingResponse(BaseModel):
    x: list[float] = Field(description="Bin centers")
    simulation: list[float] = Field(description="Simulated PDF values")
    analytical: list[float] = Field(description="Analytical PDF values")


class DoubleExponentialResponse(SamplingResponse):
    char_x: list[float] = Field(description="X values from characteristic function")
    char_y: list[float] = Field(description="Y values from characteristic function")


@sampling_router.get("/gaussian-sampling")
async def gaussian_sampling(
    kappa: float = Query(1.0, description="Mean reversion speed", ge=0.1, le=5.0),
    samples: int = Query(1000, description="Number of sample paths", ge=100, le=10000),
) -> SamplingResponse:
    pr = Vasicek(rate=0.5, kappa=kappa)
    paths = pr.sample(samples, 1, 1000)
    pdf = paths.pdf(num_bins=50)
    x = [float(v) for v in pdf.index]
    simulation = [float(v) for v in pdf["pdf"]]
    analytical = [float(v) for v in np.atleast_1d(pr.marginal(1).pdf(pdf.index))]
    return SamplingResponse(x=x, simulation=simulation, analytical=analytical)


@sampling_router.get("/poisson-sampling")
async def poisson_sampling(
    intensity: float = Query(2.0, description="Poisson intensity", ge=2.0, le=5.0),
    samples: int = Query(1000, description="Number of sample paths", ge=100, le=10000),
) -> SamplingResponse:
    pr = PoissonProcess(intensity=intensity)
    paths = pr.sample(samples, 1, 1000)
    pdf = paths.pdf(delta=1)
    x = [float(v) for v in pdf.index]
    simulation = [float(v) for v in pdf["pdf"]]
    analytical = [float(v) for v in np.atleast_1d(pr.marginal(1).pdf(pdf.index))]
    return SamplingResponse(x=x, simulation=simulation, analytical=analytical)


@sampling_router.get("/double-exponential-sampling")
async def double_exponential_sampling(
    log_kappa: float = Query(
        0.1, description="Log of asymmetry parameter", ge=-2.0, le=2.0
    ),
    samples: int = Query(1000, description="Number of samples", ge=100, le=10000),
) -> DoubleExponentialResponse:
    pr = DoubleExponential.from_moments(kappa=np.exp(log_kappa))
    data = pr.sample(samples)
    pdf = bins.pdf(data, num_bins=50, symmetric=0)
    x = [float(v) for v in pdf.index]
    simulation = [float(v) for v in pdf["pdf"]]
    analytical = [float(v) for v in np.atleast_1d(pr.pdf(pdf.index))]
    cha = pr.pdf_from_characteristic()
    char_x = [float(v) for v in cha.x]
    char_y = [float(v) for v in cha.y]
    return DoubleExponentialResponse(
        x=x,
        simulation=simulation,
        analytical=analytical,
        char_x=char_x,
        char_y=char_y,
    )
