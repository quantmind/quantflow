from typing import List

import numpy as np

from ..utils.types import Vector, as_array
from .base import Im
from .cir import IntensityProcess, Parameters
from .poisson import PoissonProcess


class DSP(PoissonProcess):
    r"""
    Doubly Stochastic Poisson process.

    It's a process where the inter-arrival time is exponentially distributed
    with rate :math:`\lambda_t`

    :param intensity: the stochastic intensity of the Poisson
    """

    def __init__(self, intensity: IntensityProcess):
        super().__init__(1)
        self.intensity = intensity

    def __repr__(self):
        return f"{type(self).__name__} {self.intensity}"

    @property
    def parameters(self) -> Parameters:
        return self.intensity.parameters

    def pdf(self, t: float, n: Vector = 0) -> Vector:
        """PDF of the number of events at time t.

        It is obtained via inverse Fourier transform of the characteristic function
        """
        nn = as_array(n)
        size = nn.shape[0]
        assert (
            size > 0 and np.log2(size).is_integer()
        ), "Must pass a power of two array of integers"
        u = -2 * np.pi * np.fft.rfftfreq(size)
        psi = self.characteristic(t, u)
        return np.fft.irfft(psi)

    def cdf(self, t: float, n: Vector) -> Vector:
        """CDF of the number of events at time t.

        It is obtained form cumulative summation of :class:`pdf`
        """
        return np.cumsum(self.pdf(t, n))

    def characteristic(self, t: float, u: Vector) -> Vector:
        phi = self.characteristic_exponent(u)
        return self.intensity.cumulative_characteristic(t, -Im * phi)

    def arrivals(self, t: float = 1) -> List[float]:
        paths = self.intensity.paths(1, t).integrate()
        return super().arrivals(paths.data[-1, 0])

    def to_proto(self):
        return self.intensity.to_proto()
