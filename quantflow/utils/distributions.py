from abc import abstractmethod
from typing import Self

import numpy as np
from pydantic import Field
from scipy import stats

from .marginal import Marginal1D
from .types import FloatArray, FloatArrayLike, Vector


class Distribution1D(Marginal1D):
    """Base class for 1D distributions to be used as
    Jump distributions in Compound Poisson processes
    """

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Sample from the distribution"""

    @classmethod
    def from_variance_and_asymmetry(cls, variance: float, asymmetry: float) -> Self:
        """Create a distribution from variance and asymmetry"""
        raise NotImplementedError

    def asymmetry(self) -> float:
        """Asymmetry of the distribution, 0 for symmetric

        Implemented by distributions that have asymmetry
        """
        raise NotImplementedError

    def set_variance(self, variance: float) -> None:
        """Set the variance of the distribution"""
        raise NotImplementedError

    def set_asymmetry(self, asymmetry: float) -> None:
        """Set the asymmetry of the distribution

        Implemented by distributions that have asymmetry
        """
        raise NotImplementedError


class Exponential(Distribution1D):
    r"""A :class:`.Distribution1D` for the `Exponential distribution`_

    The exponential distribution is a continuous probability distribution with PDF
    given by

    .. math::
        f(x) = \lambda e^{-\lambda x}\ \ \forall x \geq 0

    .. _Exponential distribution: https://en.wikipedia.org/wiki/Exponential_distribution
    """

    decay: float = Field(default=1, gt=0, description="exponential decay rate")
    r"""The exponential decay rate :math:`\lambda`"""

    @property
    def scale(self) -> float:
        """The scale parameter, it is the inverse of the :attr:`.decay` rate"""
        return 1 / self.decay

    @property
    def scale2(self) -> float:
        return self.scale**2

    def characteristic(self, u: Vector) -> Vector:
        return self.decay / (self.decay - complex(0, 1) * u)

    def mean(self) -> float:
        return self.scale

    def variance(self) -> float:
        return self.scale2

    def sample(self, n: int) -> np.ndarray:
        return np.random.exponential(scale=self.scale, size=n)

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return np.linspace(0, std_mult * np.max(self.std()), points)

    def pdf(self, x: FloatArrayLike) -> FloatArrayLike:
        """The analytical PDF of the exponential distribution as defined above"""
        return self.decay * np.exp(-self.decay * x)

    def cdf(self, x: FloatArrayLike) -> FloatArrayLike:
        r"""The analytical CDF of the exponential distribution

        .. math::
            F(x) = 1 - e^{-\lambda x}\ \ \forall x \geq 0

        """
        return 1.0 - np.exp(-self.decay * x)


class Normal(Distribution1D):
    r"""A :class:`.Distribution1D` for the `Normal distribution`_

    The normal distribution is a continuous probability distribution with PDF
    given by

    .. math::
        f(x) = \frac{e^{-\frac{\left(x - \mu\right)^2}{2\sigma^2}}}{\sqrt{2\pi\sigma^2}}

    .. _Normal distribution: https://en.wikipedia.org/wiki/Normal_distribution
    """

    mu: float = Field(default=0, description="mean")
    r"""The mean :math:`\mu` of the normal distribution"""
    sigma: float = Field(default=1, gt=0, description="standard deviation")
    r"""The standard deviation :math:`\sigma` of the normal distribution"""

    @classmethod
    def from_variance_and_asymmetry(cls, variance: float, asymmetry: float) -> Self:
        """The normal distribution is symmetric, so the asymmetry is ignored"""
        return cls(mu=0, sigma=np.sqrt(variance))

    @property
    def sigma2(self) -> float:
        return self.sigma**2

    def characteristic(self, u: Vector) -> Vector:
        return np.exp(complex(0, 1) * u * self.mu - self.sigma2 * u**2 / 2)

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma2

    def sample(self, n: int) -> np.ndarray:
        return np.random.normal(loc=self.mu, scale=self.sigma, size=n)

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return np.linspace(
            self.mu - std_mult * self.sigma,
            self.mu + std_mult * self.sigma,
            points,
        )

    def set_variance(self, variance: float) -> None:
        """Set the variance of the distribution"""
        self.sigma = np.sqrt(variance)


class DoubleExponential(Exponential):
    r"""A :class:`.Marginal1D` for the generalized double exponential distribution

    This is also know as the Asymmetric Laplace distribution (`ALD`_) which is
    a continuous probability distribution with PDF

    .. math::
        \begin{align}
            f(x) &= \frac{\lambda}{\kappa + \frac{1}{\kappa}}
                e^{-\left(x - m\right) \lambda s(x) \kappa^{s(x)}}\\
            s(x) &= {\tt sgn}\left({x - m}\right)
        \end{align}

    where `m` is the :attr:`.loc` parameter, :math:`\lambda` is the :attr:`.decay`
    parameter, and :math:`\kappa` is the asymmetric :attr:`.kappa` parameter.

    The Asymmetric Laplace distribution is similar to the Gaussian/normal distribution,
    but is sharper at the peak, it has fatter tails and allow for skewness.
    It represents the difference between two independent, exponential random variables.

    .. _ALD: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
    """

    loc: float = Field(default=0, description="location parameter")
    """The location parameter `m`"""
    decay: float = Field(default=1, gt=0, description="exponential decay rate")
    r"""The exponential decay rate :math:`\lambda`"""
    kappa: float = Field(default=1, gt=0, description="asymmetric parameter")
    """Asymmetric parameter - when k=1, the distribution is symmetric"""

    @property
    def log_kappa(self) -> float:
        """The log of the :attr:`.kappa` parameter"""
        return np.log(self.kappa)

    @classmethod
    def from_variance_and_asymmetry(cls, variance: float, asymmetry: float) -> Self:
        return cls.from_moments(variance=variance, kappa=np.exp(asymmetry))

    @classmethod
    def from_moments(
        cls,
        *,
        mean: float = 0,
        variance: float = 1,
        kappa: float = 1,
    ) -> Self:
        r"""Create a double exponential distribution from the mean, variance
        and asymmetry

        :param mean: The mean of the distribution
        :param variance: The variance of the distribution
        :param kappa: The asymmetry parameter of the distribution, 1 for symmetric
        """
        k2 = kappa * kappa
        decay = np.sqrt((1 + k2 * k2) / (variance * k2))
        return cls(decay=decay, kappa=kappa, loc=mean - (1 - k2) / (kappa * decay))

    def characteristic(self, u: Vector) -> Vector:
        r"""Characteristic function of the double exponential distribution

        .. math::
            \phi(u) = \frac{e^{i u m}}{\left(1 + \frac{i u \kappa}{\lambda}\right)
                \left(1 - \frac{i u}{\lambda \kappa}\right)}
        """
        den = (1.0 + 1j * u * self.kappa / self.decay) * (
            1.0 - 1j * u / (self.kappa * self.decay)
        )
        return np.exp(1j * u * self.loc) / den

    def mean(self) -> float:
        r"""The mean of the double exponential distribution"""
        return stats.laplace_asymmetric.mean(self.kappa, loc=self.loc, scale=self.scale)

    def variance(self) -> float:
        return stats.laplace_asymmetric.var(self.kappa, loc=self.loc, scale=self.scale)

    def pdf(self, x: FloatArrayLike) -> FloatArrayLike:
        """The analytical PDF as defined above"""
        return stats.laplace_asymmetric.pdf(
            x, self.kappa, loc=self.loc, scale=self.scale
        )

    def sample(self, n: int) -> np.ndarray:
        """Sample from the double exponential distribution"""
        return stats.laplace_asymmetric.rvs(
            self.kappa, loc=self.loc, scale=self.scale, size=n
        )

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return np.linspace(
            self.mean() - std_mult * self.std(),
            self.mean() + std_mult * self.std(),
            points,
        )

    def asymmetry(self) -> float:
        """Asymmetry of the distribution"""
        return np.log(self.kappa)

    def set_variance(self, variance: float) -> None:
        """Set the variance of the distribution"""
        k2 = self.kappa * self.kappa
        self.decay = np.sqrt((1 + k2 * k2) / (variance * k2))

    def set_asymmetry(self, asymmetry: float) -> None:
        self.kappa = np.exp(asymmetry)
