from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.optimize import Bounds

from .transforms import Transform, TransformResult, default_bounds
from .types import FloatArray, FloatArrayLike, Vector


class Marginal1D(BaseModel, ABC, extra="forbid"):
    """Marginal distribution"""

    @abstractmethod
    def characteristic(self, u: Vector) -> Vector:
        """
        Compute the characteristic function on support points `n`.
        """

    @abstractmethod
    def support(self, points: int = 100, *, std_mult: float = 3) -> FloatArray:
        """
        Compute the x axis.
        """

    def mean(self) -> FloatArrayLike:
        """Expected value

        This should be overloaded if a more efficient way of computing the mean
        """
        return self.mean_from_characteristic()

    def variance(self) -> FloatArrayLike:
        """Variance

        This should be overloaded if a more efficient way of computing the
        """
        return self.variance_from_characteristic()

    def domain_range(self) -> Bounds:
        """The space domain range for the random variable

        This should be overloaded if required
        """
        return default_bounds()

    def frequency_range(self, max_frequency: float | None = None) -> float:
        """The frequency domain range for the characteristic function

        This should be overloaded if required
        """
        return Bounds(0, max_frequency or 20)

    def std(self) -> FloatArrayLike:
        """Standard deviation at a time horizon"""
        return np.sqrt(self.variance())

    def mean_from_characteristic(self, *, d: float = 0.001) -> FloatArrayLike:
        """Calculate mean as first derivative of characteristic function at 0"""
        m = -0.5 * 1j * (self.characteristic(d) - self.characteristic(-d)) / d
        return m.real

    def std_from_characteristic(self) -> FloatArrayLike:
        """Calculate standard deviation as square root of variance"""
        return np.sqrt(self.variance_from_characteristic())

    def variance_from_characteristic(self, *, d: float = 0.001) -> FloatArrayLike:
        """Calculate variance as second derivative of characteristic function at 0"""
        c1 = self.characteristic(d)
        c0 = self.characteristic(0)
        c2 = self.characteristic(-d)
        m = -0.5 * 1j * (c1 - c2) / d
        s = -(c1 - 2 * c0 + c2) / (d * d) - m * m
        return s.real

    def cdf(self, x: FloatArrayLike) -> FloatArrayLike:
        """
        Compute the cumulative distribution function

        :param n: Location in the stochastic process domain space. If a numpy array,
            the output should have the same shape as the input.
        """
        raise NotImplementedError("Analytical CFD not available")

    def pdf(self, x: FloatArrayLike) -> FloatArrayLike:
        """
        Computes the probability density (or mass) function of the process.

        It has a base implementation that computes the pdf from the
        :class:`cdf` method, but a subclass should overload this method if a
        more optimized way of computing it is available.

        :param n: Location in the stochastic process domain space. If a numpy array,
            the output should have the same shape as the input.
        """
        raise NotImplementedError("Analytical PFD not available")

    def pdf_from_characteristic(
        self,
        n: int | None = None,
        *,
        max_frequency: float | None = None,
        simpson_rule: bool = False,
        use_fft: bool = False,
        frequency_n: int | None = None,
    ) -> TransformResult:
        """
        Compute the probability density function from the characteristic function.

        :param n: Number of discretization points to use in the transform.
            If None, use 128.
        :param max_frequency: The maximum frequency to use in the transform. If not
            provided, the value from the :meth:`frequency_range` method is used.
            Only needed for special cases/testing.
        :param simpson_rule: Use Simpson's rule for integration. Default is False.
        :param use_fft: Use FFT for the transform. Default is False.
        """
        transform = self.get_transform(
            frequency_n or n,
            self.support,
            max_frequency=max_frequency,
            simpson_rule=simpson_rule,
            use_fft=use_fft,
        )
        psi = cast(np.ndarray, self.characteristic(transform.frequency_domain))
        return transform(psi, use_fft=use_fft)

    def cdf_from_characteristic(
        self,
        n: int | None = None,
        *,
        max_frequency: float | None = None,
        simpson_rule: bool = False,
        use_fft: bool = False,
        frequency_n: int | None = None,
    ) -> TransformResult:
        raise NotImplementedError("CFD not available")

    def call_option(
        self,
        n: int | None = None,
        *,
        max_frequency: float | None = None,
        max_moneyness: float = 1,
        alpha: float | None = None,
        simpson_rule: bool = False,
        use_fft: bool = False,
    ) -> TransformResult:
        transform = self.get_transform(
            n,
            lambda m: self.option_support(m + 1, max_moneyness=max_moneyness),
            max_frequency=max_frequency,
            simpson_rule=simpson_rule,
            use_fft=use_fft,
        )
        alpha = alpha or self.option_alpha()
        phi = cast(
            np.ndarray,
            self.call_option_transform(transform.frequency_domain - 1j * alpha),
        )
        result = transform(phi, use_fft=use_fft)
        return TransformResult(x=result.x, y=result.y * np.exp(-alpha * result.x))

    def option_time_value(
        self,
        n: int = 128,
        *,
        max_frequency: float | None = None,
        max_moneyness: float = 1,
        alpha: float = 1.1,
        simpson_rule: bool = False,
        use_fft: bool = False,
    ) -> TransformResult:
        """Option time value"""
        transform = self.get_transform(
            n,
            lambda m: self.option_support(m + 1, max_moneyness=max_moneyness),
            max_frequency=max_frequency,
            simpson_rule=simpson_rule,
            use_fft=use_fft,
        )
        phi = cast(
            np.ndarray,
            self.option_time_value_transform(transform.frequency_domain, alpha),
        )
        result = transform(phi, use_fft=use_fft)
        time_value = result.y / np.sinh(alpha * result.x)
        return TransformResult(x=result.x, y=time_value)

    def characteristic_df(
        self,
        n: int | None = None,
        *,
        max_frequency: float | None = None,
        simpson_rule: bool = False,
    ) -> pd.DataFrame:
        """
        Compute the characteristic function with n discretization points
        and a max frequency
        """
        transform = self.get_transform(
            n,
            self.support,
            max_frequency=max_frequency,
            simpson_rule=simpson_rule,
        )
        psi = self.characteristic(transform.frequency_domain)
        return transform.characteristic_df(cast(np.ndarray, psi))

    def get_transform(
        self,
        n: int | None,
        support: Callable[[int], FloatArray],
        *,
        max_frequency: float | None = None,
        simpson_rule: bool = False,
        use_fft: bool = False,
    ) -> Transform:
        n = n or 128
        if use_fft:
            bounds = self.domain_range()
        else:
            x = support(n)
            bounds = Bounds(float(np.min(x)), float(np.max(x)))
        return Transform.create(
            n,
            frequency_range=self.frequency_range(max_frequency),
            domain_range=bounds,
            simpson_rule=simpson_rule,
        )

    def pdf_jacobian(self, x: FloatArrayLike) -> FloatArrayLike:
        """
        Jacobian of the pdf with respect to the parameters of the process.
        It has a base implementation that computes it from the
        :class:`cdf_jacobian` method, but a subclass should overload this method if a
        more optimized way of computing it is available.
        """
        return self.cdf_jacobian(x) - self.cdf_jacobian(x - 1)

    def cdf_jacobian(self, x: FloatArrayLike) -> np.ndarray:
        """
        Jacobian of the cdf with respect to the parameters of the process.
        It is useful for optimization purposes if necessary.

        Optional to implement, otherwise raises ``NotImplementedError`` if called.
        """
        raise NotImplementedError("Analytical CFD Jacobian not available")

    def option_support(
        self, points: int = 101, max_moneyness: float = 1.0
    ) -> FloatArray:
        """
        Compute the x axis.
        """
        return np.linspace(-max_moneyness, max_moneyness, points)

    # Fourier Transforms for options

    def call_option_transform(self, u: Vector) -> Vector:
        """Call option transform"""
        uj = 1j * u
        return self.characteristic_corrected(u - 1j) / (uj * uj + uj)

    def characteristic_corrected(self, u: Vector) -> Vector:
        convexity = np.log(self.characteristic(-1j))
        return self.characteristic(u) * np.exp(-1j * u * convexity)

    def option_time_value_transform(self, u: Vector, alpha: float = 1.1) -> Vector:
        """Option time value transform

        This transform does not require any additional correction since
        the integrant is already bounded for positive and negative moneyess"""
        ia = 1j * alpha
        return 0.5 * (
            self._option_time_value_transform(u - ia)
            - self._option_time_value_transform(u + ia)
        )

    def _option_time_value_transform(self, u: Vector) -> Vector:
        """Option time value transform

        This transform does not require any additional correction since
        the integrant is already bounded for positive and negative moneyess"""
        iu = 1j * u
        return (
            1 / (1 + iu) - 1 / iu - self.characteristic_corrected(u - 1j) / (u * u - iu)
        )

    def option_alpha(self) -> float:
        """Option alpha"""
        return 2.0
