from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.optimize import Bounds

from .transforms import Transform, TransformResult, default_bounds
from .types import FloatArray, Vector, FloatArrayLike


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

        THis should be overloaded if a more efficient way of computing the mean
        """
        return self.mean_from_characteristic()

    def variance(self) -> FloatArrayLike:
        """Variance

        This should be overloaded if a more efficient way of computing the
        """
        return self.variance_from_characteristic()

    def max_frequency(self) -> float:
        """Maximum frequency of the characteristic function

        This should be overloaded if required
        """
        return 20

    def std(self) -> FloatArrayLike:
        """Standard deviation at a time horizon"""
        return np.sqrt(self.variance())

    def mean_from_characteristic(self) -> FloatArrayLike:
        """Calculate mean as first derivative of characteristic function at 0"""
        d = 0.001
        m = -0.5 * 1j * (self.characteristic(d) - self.characteristic(-d)) / d
        return cast(complex, m).real

    def std_from_characteristic(self) -> FloatArrayLike:
        """Calculate standard deviation as square root of variance"""
        return np.sqrt(self.variance_from_characteristic())

    def variance_from_characteristic(self) -> FloatArrayLike:
        """Calculate variance as second derivative of characteristic function at 0"""
        d = 0.001
        c1 = self.characteristic(d)
        c0 = self.characteristic(0)
        c2 = self.characteristic(-d)
        m = -0.5 * 1j * (c1 - c2) / d
        s = -(c1 - 2 * c0 + c2) / (d * d) - m * m
        return s.real

    def pdf(self, x: FloatArrayLike) -> FloatArrayLike:
        """
        Computes the probability density (or mass) function of the process.

        It has a base implementation that computes the pdf from the
        :class:`cdf` method, but a subclass should overload this method if a
        more optimized way of computing it is available.

        :param n: Location in the stochastic process domain space. If a numpy array,
            the output should have the same shape as the input.
        """
        return self.cdf(x) - self.cdf(x - 1)

    def pdf_from_characteristic(
        self,
        n: int | None = None,
        *,
        max_frequency: float | None = None,
        simpson_rule: bool = False,
        use_fft: bool = False,
    ) -> TransformResult:
        """
        Compute the probability density function from the characteristic function.
        """
        n = n or 128
        if use_fft:
            delta_x = None
            transform = Transform(
                n,
                max_frequency=self.get_max_frequency(max_frequency),
                domain_range=self.domain_range(),
                simpson_rule=simpson_rule,
            )
        else:
            x = self.support(n + 1)
            min_x = float(np.min(x))
            max_x = float(np.max(x))
            delta_x = (max_x - min_x) / (len(x) - 1)
            transform = Transform(
                n,
                max_frequency=self.get_max_frequency(max_frequency),
                domain_range=Bounds(min_x, max_x),
                simpson_rule=simpson_rule,
            )
        psi = cast(np.ndarray, self.characteristic(transform.frequency_domain))
        return transform(psi, delta_x)

    def call_option(
        self,
        n: int | None = None,
        *,
        max_frequency: float | None = None,
        max_moneyness: float = 1,
        alpha: float | None = None,
        simpson_rule: bool = False,
    ) -> TransformResult:
        n = n or 128
        x = self.option_support(n + 1, max_moneyness=max_moneyness)
        min_x = float(np.min(x))
        max_x = float(np.max(x))
        delta_x = (max_x - min_x) / (len(x) - 1)
        transform = Transform(
            n,
            max_frequency=self.get_max_frequency(max_frequency),
            domain_range=Bounds(min_x, max_x),
            simpson_rule=simpson_rule,
        )
        alpha = alpha or self.option_alpha()
        phi = cast(
            np.ndarray,
            self.call_option_transform(transform.frequency_domain - 1j * alpha),
        )
        result = transform(phi, delta_x)
        return TransformResult(x=result.x, y=result.y * np.exp(-alpha * result.x))

    def option_time_value(
        self,
        n: int = 128,
        *,
        max_frequency: float | None = None,
        max_moneyness: float = 1,
        alpha: float = 1.1,
        simpson_rule: bool = False,
    ) -> TransformResult:
        """Option time value"""
        n = n or 128
        x = self.option_support(n + 1, max_moneyness=max_moneyness)
        min_x = float(np.min(x))
        max_x = float(np.max(x))
        delta_x = (max_x - min_x) / (len(x) - 1)
        transform = Transform(
            n,
            max_frequency=self.get_max_frequency(max_frequency),
            domain_range=Bounds(min_x, max_x),
            simpson_rule=simpson_rule,
        )
        phi = cast(
            np.ndarray,
            self.option_time_value_transform(transform.frequency_domain, alpha),
        )
        result = transform(phi, delta_x)
        time_value = result.y / np.sinh(alpha * result.x)
        return TransformResult(x=result.x, y=time_value)

    def domain_range(self) -> Bounds:
        return default_bounds()

    def cdf(self, x: FloatArrayLike) -> FloatArrayLike:
        """
        Compute the cumulative distribution function

        :param n: Location in the stochastic process domain space. If a numpy array,
            the output should have the same shape as the input.
        """
        raise NotImplementedError("Analytical CFD not available")

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

    def characteristic_df(
        self, n: int | None, max_frequency: float | None = None, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Compute the characteristic function with n discretization points
        and a max frequency
        """
        fre = Transform(
            n=n, max_frequency=self.get_max_frequency(max_frequency), **kwargs
        ).frequency_domain
        psi = self.characteristic(fre)
        return pd.concat(
            (
                pd.DataFrame(dict(frequency=fre, characteristic=psi.real, name="real")),
                pd.DataFrame(dict(frequency=fre, characteristic=psi.imag, name="iamg")),
            )
        )

    def get_max_frequency(self, max_frequency: float | None = None) -> float:
        """
        Get the maximum frequency to use for the characteristic function
        """
        return max_frequency if max_frequency is not None else self.max_frequency()

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
