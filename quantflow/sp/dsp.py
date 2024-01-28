import math

import numpy as np
from pydantic import Field
from scipy.optimize import Bounds

from ..utils.types import FloatArray, FloatArrayLike, Vector
from .base import StochasticProcess1DMarginal
from .cir import CIR, IntensityProcess
from .poisson import MarginalDiscrete1D, PoissonBase, PoissonProcess, poisson_arrivals


class DSP(PoissonBase):
    r"""
    Doubly Stochastic Poisson process.

    It's a process where the inter-arrival time is exponentially distributed
    with rate :math:`\lambda_t`

    :param intensity: the stochastic intensity of the Poisson
    """

    intensity: IntensityProcess = Field(  # type ignore
        default_factory=CIR, description="intensity process"
    )
    poisson: PoissonProcess = Field(default_factory=PoissonProcess, exclude=True)

    def marginal(self, t: FloatArrayLike) -> StochasticProcess1DMarginal:
        return MarginalDiscrete1D(process=self, t=t)

    def frequency_range(self, std: float, max_frequency: float | None = None) -> Bounds:
        """Frequency range of the process"""
        return Bounds(0, np.pi)

    def support(self, mean: float, std: float, points: int) -> FloatArray:
        return np.linspace(0, points, points + 1)

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        phi = self.poisson.characteristic_exponent(t, u)
        return -self.intensity.integrated_log_laplace(t, phi)

    def arrivals(self, t: float = 1) -> list[float]:
        paths = self.intensity.sample(1, t, math.ceil(100 * t)).integrate()
        intensity = paths.data[-1, 0]
        return poisson_arrivals(intensity, t)

    def sample_jumps(self, n: int) -> FloatArray:
        return self.poisson.sample_jumps(n)
