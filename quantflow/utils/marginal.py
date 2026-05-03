from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Callable, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.optimize import Bounds
from typing_extensions import Annotated, Doc

from .transforms import Transform, TransformResult, default_bounds
from .types import FloatArray, FloatArrayLike, Vector


class OptionPricingMethod(enum.StrEnum):
    """Method to use for option pricing via Fourier transform of the call option
    price"""

    CARR_MADAN = enum.auto()
    LEWIS = enum.auto()
    COS = enum.auto()


class OptionPricingResult(BaseModel, ABC):
    method: OptionPricingMethod

    @abstractmethod
    def call_at(self, log_strikes: FloatArrayLike) -> FloatArray:
        """Evaluate call prices at arbitrary log-strikes."""


class OptionPricingTransformResult(OptionPricingResult, arbitrary_types_allowed=True):
    log_strikes: FloatArray
    call_prices: FloatArray

    def call_at(self, log_strikes: FloatArrayLike) -> FloatArray:
        return cast(
            FloatArray,
            np.interp(log_strikes, self.log_strikes, self.call_prices),
        )


class OptionPricingCosResult(OptionPricingResult, arbitrary_types_allowed=True):
    """Result of call option pricing via the COS method.

    Stores the precomputed coefficient vector, enabling $O(N)$ evaluation
    at any log-strike without recomputing the characteristic function.
    """

    a: float = Field(description="Left endpoint of the truncation interval [a, b]")
    nu: FloatArray = Field(description="Cosine frequency grid j*pi/(b-a)")
    coeff: np.ndarray = Field(
        description="Complex coefficient vector w_j * phi(nu_j) * V_j"
    )

    def call_at(
        self,
        log_strikes: Annotated[
            FloatArrayLike, Doc("Log-strikes at which to evaluate the call price")
        ],
    ) -> FloatArray:
        """Evaluate call prices at arbitrary log-strikes in $O(N)$ per strike."""
        k = np.asarray(log_strikes)
        return np.real(
            np.exp(-1j * np.outer(k + self.a, self.nu)) @ self.coeff
        ) * np.exp(k)


class Marginal1D(BaseModel, ABC, extra="forbid"):
    r"""Abstract 1D marginal distribution with Fourier-based option pricing.

    This class represents the marginal distribution.
    It provides methods to compute the pdf, cdf, and option
    prices from the characteristic function of the underlying process, as well as
    their Jacobians with respect to the parameters of the process.

    Option prices are computed via Fourier inversion of the
    [characteristic function](/glossary#characteristic-function),
    using two supported formulas:

    * [Carr & Madan](/bibliography#carr_madan): uses a damping parameter $\alpha$
      to ensure integrability.
      The integrand is evaluated along a contour shifted by $\alpha$ in the
      imaginary direction. See [call_option_carr_madan][.call_option_carr_madan].
    * [Lewis](/bibliography#lewis): no damping parameter required.
      The contour is fixed at
      imaginary part $1/2$, giving an integrand that is naturally bounded for
      all real $u$. See [call_option_lewis][.call_option_lewis].
    * [COS method](/bibliography#cos): uses a Fourier-cosine expansion of the
      option payoff, with coefficients that depend on the characteristic function
      evaluated at a cosine frequency grid. See [call_option_cos][.call_option_cos].

    It is the base class for the
    [StochasticProcess1DMarginal][quantflow.sp.base.StochasticProcess1DMarginal].
    """

    def call_option(
        self,
        n: Annotated[
            int | None,
            Doc("Number of discretization points for the transform. Defaults to 128."),
        ] = None,
        *,
        pricing_method: Annotated[
            OptionPricingMethod,
            Doc(
                "Method to use for option pricing via Fourier transform of the "
                "call option price. Defaults to Lewis."
            ),
        ] = OptionPricingMethod.CARR_MADAN,
        max_frequency: Annotated[
            float | None,
            Doc(
                "Maximum frequency for the transform grid. "
                "Defaults to frequency_range()."
            ),
        ] = None,
        max_log_strike: Annotated[
            float, Doc("Maximum absolute log-strike for the output grid.")
        ] = 1,
        alpha: Annotated[
            float | None,
            Doc(
                "Damping parameter for integrability of the Carr-Madan integrand, "
                "it is ignored if not applicable."
            ),
        ] = None,
        simpson_rule: Annotated[
            bool, Doc("Use Simpson's rule for integration. Default is False.")
        ] = False,
        use_fft: Annotated[
            bool, Doc("Use FFT for the transform rather than FRFT. Default is False.")
        ] = False,
    ) -> OptionPricingResult:
        """Price call options via one of the available Fourier-based methods,
        as a function of log-strike
        """
        match pricing_method:
            case OptionPricingMethod.COS:
                return self.call_option_cos(
                    n,
                    max_log_strike=max_log_strike,
                )
            case OptionPricingMethod.CARR_MADAN:
                return self.call_option_carr_madan(
                    n,
                    max_frequency=max_frequency,
                    max_log_strike=max_log_strike,
                    alpha=alpha,
                    simpson_rule=simpson_rule,
                    use_fft=use_fft,
                )
            case _:
                return self.call_option_lewis(
                    n,
                    max_frequency=max_frequency,
                    max_log_strike=max_log_strike,
                    simpson_rule=simpson_rule,
                    use_fft=use_fft,
                )

    def call_option_carr_madan(
        self,
        n: Annotated[
            int | None,
            Doc("Number of discretization points for the transform. Defaults to 128."),
        ] = None,
        *,
        max_frequency: Annotated[
            float | None,
            Doc(
                "Maximum frequency for the transform grid. "
                "Defaults to frequency_range()."
            ),
        ] = None,
        max_log_strike: Annotated[
            float, Doc("Maximum absolute log-strike for the output grid.")
        ] = 1,
        alpha: Annotated[
            float | None,
            Doc(
                "Damping parameter for integrability of the Carr-Madan integrand. "
                "Defaults to "
                "[call_option_carr_madan_alpha][..call_option_carr_madan_alpha]."
            ),
        ] = None,
        simpson_rule: Annotated[
            bool, Doc("Use Simpson's rule for integration. Default is False.")
        ] = False,
        use_fft: Annotated[
            bool, Doc("Use FFT for the transform. Default is False.")
        ] = False,
    ) -> OptionPricingTransformResult:
        """Call option price via Carr & Madan method"""
        transform = self.get_transform(
            n,
            lambda m: self.option_support(m + 1, max_log_strike=max_log_strike),
            max_frequency=max_frequency,
            simpson_rule=simpson_rule,
            use_fft=use_fft,
        )
        alpha = alpha or self.call_option_carr_madan_alpha()
        phi = cast(
            np.ndarray,
            self.call_option_transform(transform.frequency_domain - 1j * alpha),
        )
        result = transform(phi, use_fft=use_fft)
        return OptionPricingTransformResult(
            method=OptionPricingMethod.CARR_MADAN,
            log_strikes=result.x,
            call_prices=result.y * np.exp(-alpha * result.x),
        )

    def call_option_carr_madan_alpha(self) -> float:
        """Option alpha to use for Carr & Madan transform to ensure integrability
        of the call option transform"""
        return 2.0

    def call_option_cos(
        self,
        n: Annotated[
            int | None,
            Doc("Number of cosine series terms. Defaults to 128."),
        ] = None,
        *,
        max_log_strike: Annotated[
            float, Doc("Maximum absolute log-strike for the output grid.")
        ] = 1,
        L: Annotated[
            float,
            Doc(
                "Truncation parameter: the integration interval is set to "
                "[-L*std, L*std]."
            ),
        ] = 12.0,
    ) -> OptionPricingCosResult:
        r"""Call option price via the COS method (Fang & Oosterlee 2008).

        The call price at log-strike $k$ is approximated by the cosine series

        \begin{equation}
            C(k) \approx e^k \sum_{j=0}^{N-1}{}^{\prime}
                \mathrm{Re}\!\left[
                    \phi\!\left(\frac{j\pi}{b-a}\right)
                    e^{-i j\pi (k+a)/(b-a)}
                \right] V_j
        \end{equation}

        where the prime denotes a half weight on the $j=0$ term, $[a,b]$ is the
        truncation interval, and $V_j$ are the cosine payoff coefficients for the
        normalised call payoff $(e^y - 1)^+$ integrated over $[0, b]$.
        The $e^k$ factor converts from strike-normalised to forward-space pricing.

        Returns an
        [OptionPricingCosResult][quantflow.utils.marginal.OptionPricingCosResult]
        with the precomputed coefficient vector. Use
        [call_at][quantflow.utils.marginal.OptionPricingCosResult.call_at]
        to evaluate at arbitrary log-strikes in $O(N)$ per strike.
        """
        n = n or 128
        std = float(self.std())
        a = -L * std
        b = L * std
        bma = b - a

        j = np.arange(n)
        nu = j * np.pi / bma

        phi = np.asarray(self.characteristic_corrected(nu))

        sin_nu_a = np.sin(nu * a)
        cos_nu_a = np.cos(nu * a)
        sign_j = (-1.0) ** j
        safe_nu = np.where(j == 0, 1.0, nu)
        chi = (np.exp(b) * sign_j - cos_nu_a + nu * sin_nu_a) / np.where(
            j == 0, 1.0, 1.0 + nu**2
        )
        psi = np.where(j == 0, b, sin_nu_a / safe_nu)
        V = 2.0 / bma * (chi - psi)
        weights = np.ones(n)
        weights[0] = 0.5
        return OptionPricingCosResult(
            method=OptionPricingMethod.COS,
            a=a,
            nu=nu,
            coeff=weights * phi * V,
        )

    def call_option_lewis(
        self,
        n: Annotated[
            int | None,
            Doc("Number of discretization points for the transform. Defaults to 128."),
        ] = None,
        *,
        max_frequency: Annotated[
            float | None,
            Doc(
                "Maximum frequency for the transform grid. "
                "Defaults to frequency_range()."
            ),
        ] = None,
        max_log_strike: Annotated[
            float, Doc("Maximum absolute log-strike for the output grid.")
        ] = 1,
        simpson_rule: Annotated[
            bool, Doc("Use Simpson's rule for integration. Default is False.")
        ] = False,
        use_fft: Annotated[
            bool, Doc("Use FFT for the transform. Default is False.")
        ] = False,
    ) -> OptionPricingTransformResult:
        """Call option price via the Lewis (2001) formula"""
        transform = self.get_transform(
            n,
            lambda m: self.option_support(m + 1, max_log_strike=max_log_strike),
            max_frequency=max_frequency,
            simpson_rule=simpson_rule,
            use_fft=use_fft,
        )
        phi = cast(np.ndarray, self.lewis_transform(transform.frequency_domain))
        result = transform(phi, use_fft=use_fft)
        k = result.x
        return OptionPricingTransformResult(
            log_strikes=k,
            call_prices=1.0 - np.exp(0.5 * k) * result.y,
            method=OptionPricingMethod.LEWIS,
        )

    def call_option_transform(
        self,
        u: Annotated[
            Vector, Doc("Frequency domain points (possibly complex-shifted).")
        ],
    ) -> Vector:
        """Call option transform"""
        uj = 1j * u
        return self.characteristic_corrected(u - 1j) / (uj * uj + uj)

    def cdf(
        self,
        x: Annotated[
            FloatArrayLike,
            Doc(
                "Location in the stochastic process domain space. If a numpy array,"
                " the output should have the same shape as the input."
            ),
        ],
    ) -> FloatArrayLike:
        """
        Compute the cumulative distribution function
        """
        raise NotImplementedError("Analytical CDF not available")

    def cdf_from_characteristic(
        self,
        n: Annotated[
            int | None,
            Doc("Number of discretization points for the transform. Defaults to 128."),
        ] = None,
        *,
        max_frequency: Annotated[
            float | None,
            Doc(
                "Maximum frequency for the transform grid. "
                "Defaults to frequency_range()."
            ),
        ] = None,
        simpson_rule: Annotated[
            bool, Doc("Use Simpson's rule for integration. Default is False.")
        ] = False,
        use_fft: Annotated[
            bool, Doc("Use FFT for the transform. Default is False.")
        ] = False,
        frequency_n: Annotated[
            int | None,
            Doc("Number of points for the frequency grid. Overrides n if provided."),
        ] = None,
    ) -> TransformResult:
        raise NotImplementedError("CDF not available")

    def cdf_jacobian(
        self,
        x: Annotated[
            FloatArrayLike, Doc("Location in the state space of the process.")
        ],
    ) -> np.ndarray:
        """
        Jacobian of the cdf with respect to the parameters of the process.
        It is useful for optimization purposes if necessary.

        Optional to implement, otherwise raises ``NotImplementedError`` if called.
        """
        raise NotImplementedError("Analytical CDF Jacobian not available")

    @abstractmethod
    def characteristic(
        self,
        u: Annotated[Vector, Doc("Frequency domain points.")],
    ) -> Vector:
        """
        Compute the characteristic function on frequency domain points $u$
        """

    def characteristic_corrected(
        self,
        u: Annotated[Vector, Doc("Frequency domain points.")],
    ) -> Vector:
        """Characteristic function corrected for the convexity of the log-price
        distribution"""
        convexity = np.log(self.characteristic(-1j))
        return self.characteristic(u) * np.exp(-1j * u * convexity)

    def characteristic_df(
        self,
        n: Annotated[
            int | None,
            Doc("Number of discretization points for the transform. Defaults to 128."),
        ] = None,
        *,
        max_frequency: Annotated[
            float | None,
            Doc(
                "Maximum frequency for the transform grid. "
                "Defaults to frequency_range()."
            ),
        ] = None,
        simpson_rule: Annotated[
            bool, Doc("Use Simpson's rule for integration. Default is False.")
        ] = False,
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

    def domain_range(self) -> Bounds:
        """The space domain range for the random variable

        This should be overloaded if required
        """
        return default_bounds()

    def frequency_range(
        self,
        max_frequency: Annotated[
            float | None,
            Doc("Upper bound of the frequency grid. Defaults to 20 if None."),
        ] = None,
    ) -> float:
        """The frequency domain range for the characteristic function

        This should be overloaded if required
        """
        return Bounds(0, max_frequency or 20)

    def get_transform(
        self,
        n: Annotated[
            int | None,
            Doc("Number of discretization points. Defaults to 128."),
        ],
        support: Annotated[
            Callable[[int], FloatArray],
            Doc("Function returning the space domain grid given the number of points."),
        ],
        *,
        max_frequency: Annotated[
            float | None,
            Doc(
                "Maximum frequency for the transform grid. "
                "Defaults to frequency_range()."
            ),
        ] = None,
        simpson_rule: Annotated[
            bool, Doc("Use Simpson's rule for integration. Default is False.")
        ] = False,
        use_fft: Annotated[
            bool, Doc("Use FFT for the transform. Default is False.")
        ] = False,
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

    def lewis_transform(
        self,
        u: Annotated[Vector, Doc("Frequency domain points.")],
    ) -> Vector:
        """Lewis (2001) call option transform - no damping parameter required"""
        return self.characteristic_corrected(u - 0.5j) / (u * u + 0.25)

    def mean(self) -> FloatArrayLike:
        """Expected value

        By default it uses the
        [mean_from_characteristic][..mean_from_characteristic] method.
        This should be overloaded if a more efficient/analytical way of computing
        the mean is available.
        """
        return self.mean_from_characteristic()

    def mean_from_characteristic(
        self,
        *,
        d: Annotated[
            float,
            Doc("Step size for finite-difference approximation of the derivative."),
        ] = 0.001,
    ) -> FloatArrayLike:
        """Calculate mean as first derivative of characteristic function at 0"""
        m = -0.5 * 1j * (self.characteristic(d) - self.characteristic(-d)) / d
        return m.real

    def option_support(
        self,
        points: Annotated[int, Doc("Number of support points.")] = 101,
        max_log_strike: Annotated[float, Doc("Maximum absolute log-strike.")] = 1.0,
    ) -> FloatArray:
        """
        Compute the x axis.
        """
        return np.linspace(-max_log_strike, max_log_strike, points)

    def option_time_value(
        self,
        n: Annotated[
            int,
            Doc("Number of discretization points for the transform."),
        ] = 128,
        *,
        max_frequency: Annotated[
            float | None,
            Doc(
                "Maximum frequency for the transform grid. "
                "Defaults to frequency_range()."
            ),
        ] = None,
        max_log_strike: Annotated[
            float, Doc("Maximum absolute log-strike for the output grid.")
        ] = 1,
        alpha: Annotated[
            float,
            Doc("Contour shift parameter controlling the integration strip."),
        ] = 1.1,
        simpson_rule: Annotated[
            bool, Doc("Use Simpson's rule for integration. Default is False.")
        ] = False,
        use_fft: Annotated[
            bool, Doc("Use FFT for the transform. Default is False.")
        ] = False,
    ) -> TransformResult:
        """Option time value"""
        transform = self.get_transform(
            n,
            lambda m: self.option_support(m + 1, max_log_strike=max_log_strike),
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

    def option_time_value_transform(
        self,
        u: Annotated[Vector, Doc("Frequency domain points.")],
        alpha: Annotated[
            float, Doc("Contour shift parameter controlling the integration strip.")
        ] = 1.1,
    ) -> Vector:
        """Option time value transform

        This transform does not require any additional correction since
        the integrand is already bounded for positive and negative moneyness"""
        ia = 1j * alpha
        return 0.5 * (
            self._option_time_value_transform(u - ia)
            - self._option_time_value_transform(u + ia)
        )

    def pdf(
        self,
        x: Annotated[
            FloatArrayLike,
            Doc(
                "Location in the stochastic process domain space. If a numpy array,"
                " the output should have the same shape as the input."
            ),
        ],
    ) -> FloatArrayLike:
        """
        Computes the probability density (or mass) function of the process.

        It has a base implementation that computes the pdf from the
        [cdf][quantflow.utils.marginal.Marginal1D.cdf] method, but a subclass should
        overload this method if a
        more optimized way of computing it is available.
        """
        raise NotImplementedError("Analytical PDF not available")

    def pdf_from_characteristic(
        self,
        n: Annotated[
            int | None,
            Doc(
                "Number of discretization points to use in the transform."
                " If None, use 128."
            ),
        ] = None,
        *,
        max_frequency: Annotated[
            float | None,
            Doc(
                "The maximum frequency to use in the transform. If not provided,"
                " the value from the [frequency_range]"
                "[quantflow.utils.marginal.Marginal1D.frequency_range] method is used."
                " Only needed for special cases/testing."
            ),
        ] = None,
        simpson_rule: Annotated[
            bool, Doc("Use Simpson's rule for integration. Default is False.")
        ] = False,
        use_fft: Annotated[
            bool, Doc("Use FFT for the transform. Default is False.")
        ] = False,
        frequency_n: Annotated[
            int | None,
            Doc("Number of points for the frequency grid. Overrides n if provided."),
        ] = None,
    ) -> TransformResult:
        """
        Compute the probability density function from the characteristic function.
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

    def pdf_jacobian(
        self,
        x: Annotated[
            FloatArrayLike, Doc("Location in the state space of the process.")
        ],
    ) -> FloatArrayLike:
        """
        Jacobian of the pdf with respect to the parameters of the process.
        It has a base implementation that computes it from the
        [cdf_jacobian][quantflow.utils.marginal.Marginal1D.cdf_jacobian] method,
        but a subclass should overload this method if a
        more optimized way of computing it is available.
        """
        return self.cdf_jacobian(x) - self.cdf_jacobian(x - 1)

    def std(self) -> FloatArrayLike:
        """Standard deviation at a time horizon"""
        return np.sqrt(self.variance())

    def std_from_characteristic(self) -> FloatArrayLike:
        """Calculate standard deviation as square root of variance"""
        return np.sqrt(self.variance_from_characteristic())

    @abstractmethod
    def support(
        self,
        points: Annotated[int, Doc("Number of support points.")] = 100,
        *,
        std_mult: Annotated[
            float, Doc("Standard deviation multiplier for the support range.")
        ] = 3,
    ) -> FloatArray:
        """
        Compute the x axis.
        """

    def variance(self) -> FloatArrayLike:
        """Variance

        By default it uses the
        [variance_from_characteristic][..variance_from_characteristic] method.
        This should be overloaded if a more efficient/analytical way of computing
        the variance is available.
        """
        return self.variance_from_characteristic()

    def variance_from_characteristic(
        self,
        *,
        d: Annotated[
            float,
            Doc("Step size for finite-difference approximation of the derivative."),
        ] = 0.001,
    ) -> FloatArrayLike:
        """Calculate variance as second derivative of characteristic function at 0"""
        c1 = self.characteristic(d)
        c0 = self.characteristic(0)
        c2 = self.characteristic(-d)
        m = -0.5 * 1j * (c1 - c2) / d
        s = -(c1 - 2 * c0 + c2) / (d * d) - m * m
        return s.real

    def _option_time_value_transform(
        self,
        u: Annotated[Vector, Doc("Frequency domain points (complex-shifted).")],
    ) -> Vector:
        """Option time value transform

        This transform does not require any additional correction since
        the integrand is already bounded for positive and negative moneyness"""
        iu = 1j * u
        return (
            1 / (1 + iu) - 1 / iu - self.characteristic_corrected(u - 1j) / (u * u - iu)
        )
