import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from scipy.stats import chisquare, ks_1samp

from app.api.docs import load_description
from quantflow.sp.ou import Vasicek
from quantflow.sp.poisson import PoissonProcess
from quantflow.ta.paths import Paths
from quantflow.utils import bins
from quantflow.utils.distributions import DoubleExponential

sampling_router = APIRouter()


class SamplingResponse(BaseModel):
    x: list[float] = Field(description="Bin centers")
    simulation: list[float] = Field(description="Simulated PDF values")
    analytical: list[float] = Field(description="Analytical PDF values")


class GaussianSamplingResponse(SamplingResponse):
    ks_statistic: float = Field(
        description="Kolmogorov-Smirnov statistic vs analytical CDF"
    )
    ks_pvalue: float = Field(description="Kolmogorov-Smirnov p-value")


class PoissonSamplingResponse(SamplingResponse):
    chi2_statistic: float = Field(description="Chi-squared goodness-of-fit statistic")
    chi2_pvalue: float = Field(description="Chi-squared p-value")


class DoubleExponentialResponse(SamplingResponse):
    char_x: list[float] = Field(description="X values from characteristic function")
    char_y: list[float] = Field(description="Y values from characteristic function")


@sampling_router.get(
    "/gaussian-sampling",
    summary="Gaussian process sampling vs analytical PDF",
    description=load_description("gaussian_sampling.md"),
)
async def gaussian_sampling(
    kappa: float = Query(1.0, description="Mean reversion speed", ge=0.1, le=5.0),
    samples: int = Query(1000, description="Number of sample paths", ge=100, le=10000),
    antithetic: bool = Query(
        True, description="Use antithetic variates variance reduction"
    ),
) -> GaussianSamplingResponse:
    pr = Vasicek(rate=0.5, kappa=kappa)
    draws = Paths.normal_draws(samples, 1, 1000, antithetic_variates=antithetic)
    paths = pr.sample_from_draws(draws)
    pdf = paths.pdf(num_bins=50)
    x = [float(v) for v in pdf.index]
    simulation = [float(v) for v in pdf["pdf"]]
    analytical = [float(v) for v in np.atleast_1d(pr.marginal(1).pdf(pdf.index))]
    final_values = paths.data[-1, :]
    ks = ks_1samp(final_values, lambda v: pr.analytical_cdf(1, v))
    return GaussianSamplingResponse(
        x=x,
        simulation=simulation,
        analytical=analytical,
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
    )


@sampling_router.get(
    "/poisson-sampling",
    summary="Poisson process sampling vs analytical PMF",
    description=load_description("poisson_sampling.md"),
)
async def poisson_sampling(
    intensity: float = Query(2.0, description="Poisson intensity", ge=2.0, le=20.0),
    samples: int = Query(1000, description="Number of sample paths", ge=100, le=10000),
) -> PoissonSamplingResponse:
    pr = PoissonProcess(intensity=intensity)
    paths = pr.sample(samples, 1, 1000)
    pdf = paths.pdf(delta=1)
    x = [float(v) for v in pdf.index]
    simulation = [float(v) for v in pdf["pdf"]]
    analytical = [float(v) for v in np.atleast_1d(pr.marginal(1).pdf(pdf.index))]
    f_obs = np.array(simulation) * samples
    f_exp = np.array(analytical) * samples
    f_exp = f_exp * (f_obs.sum() / f_exp.sum())
    # merge tail bins with expected count < 5 to satisfy chi-squared requirements
    while len(f_exp) > 1 and f_exp[0] < 5:
        f_obs[1] += f_obs[0]
        f_exp[1] += f_exp[0]
        f_obs, f_exp = f_obs[1:], f_exp[1:]
    while len(f_exp) > 1 and f_exp[-1] < 5:
        f_obs[-2] += f_obs[-1]
        f_exp[-2] += f_exp[-1]
        f_obs, f_exp = f_obs[:-1], f_exp[:-1]
    chi2 = chisquare(f_obs, f_exp)
    return PoissonSamplingResponse(
        x=x,
        simulation=simulation,
        analytical=analytical,
        chi2_statistic=float(chi2.statistic),
        chi2_pvalue=float(chi2.pvalue),
    )


@sampling_router.get(
    "/double-exponential-sampling",
    summary="Double exponential sampling vs analytical PDF",
    description=load_description("double_exponential_sampling.md"),
)
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
