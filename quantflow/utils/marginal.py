from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from .frft import frft
from .types import Vector


def trapezoid(N: int) -> np.ndarray:
    h = np.ones(N)
    h[0] = 0.5
    return h


def simpson(N: int) -> np.ndarray:
    h = np.ones(N)
    h[1::2] = 4
    h[2::2] = 2
    return h / 3


class Marginal1D(ABC):
    def mean(self) -> float:
        """Expected value at a time horizon"""
        return self.mean_from_characteristic()

    def std(self) -> float:
        """Standard deviation at a time horizon"""
        return np.sqrt(self.variance())

    def variance(self) -> float:
        """Variance at a time horizon"""
        return self.variance_from_characteristic()

    def mean_from_characteristic(self) -> float:
        """Calculate mean as first derivative of characteristic function at 0"""
        d = 0.001
        m = -0.5 * 1j * (self.characteristic(d) - self.characteristic(-d)) / d
        return m.real

    def variance_from_characteristic(self) -> float:
        """Calculate variance as second derivative of characteristic function at 0"""
        d = 0.001
        c1 = self.characteristic(d)
        c0 = self.characteristic(0)
        c2 = self.characteristic(-d)
        m = -0.5 * 1j * (c1 - c2) / d
        s = -(c1 - 2 * c0 + c2) / (d * d) - m * m
        return s.real

    def pdf(self, n: Vector) -> Vector:
        """
        Computes the probability density (or mass) function of the process.

        It has a base implementation that computes the pdf from the
        :class:`cdf` method, but a subclass should overload this method if a
        more optimized way of computing it is available.

        :param t: time horizon
        :param n: Location in the stochastic process domain space. If a numpy array,
            the output should have the same shape as the input.
        """
        return self.cdf(n) - self.cdf(n - 1)

    def frequency_space(self, N: int, max_frequency: float = 10.0) -> np.ndarray:
        return 2 * max_frequency * np.fft.rfftfreq(N)

    def pdf_from_fft(
        self, N: int, max_frequency: float = 10.0, simpson_rule: bool = False
    ) -> np.ndarray:
        h = simpson(N // 2 + 1) if simpson_rule else trapezoid(N // 2 + 1)
        freq = self.frequency_space(N, max_frequency)
        dx = np.pi / max_frequency
        b = N * dx / 2
        f = (
            max_frequency
            * h
            * np.exp(1j * freq * b)
            * self.characteristic(freq)
            / np.pi
        )
        y = np.fft.irfft(f)
        return pd.DataFrame(dict(x=dx * np.linspace(0, N, N + 1)[:-1] - b, y=y))

    def pdf_from_frft(
        self, N: int, max_frequency: float, domain: Optional[float] = None
    ) -> np.ndarray:
        """Calculate the PDF using the fractional FFT"""
        g = np.arange(0, N, 1.0)
        delta_u = max_frequency / N
        if domain:
            delta_x = domain / N
        else:
            domain = 2 * np.pi / delta_u
            delta_x = domain / N
        zeta = delta_u * delta_x
        b = 0.5 * domain
        u = delta_u * g
        f = (
            N
            * delta_u
            * trapezoid(N)
            * np.exp(1j * u * b)
            * self.characteristic(u)
            / np.pi
        )
        return frft.calculate(f, zeta)

    @abstractmethod
    def cdf(self, n: Vector) -> Vector:
        """
        Compute the cumulative distribution function of the process.

        :param t: time horizon
        :param n: Location in the stochastic process domain space. If a numpy array,
            the output should have the same shape as the input.
        """

    @abstractmethod
    def characteristic(self, n: Vector) -> Vector:
        """
        Compute the characteristic function on support points `n`.
        """


def frft_coef(zeta: float, g: np.ndarray) -> np.ndarray:
    return np.exp(1j * np.pi * g * g * zeta)
