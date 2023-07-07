from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds


class TransformError(RuntimeError):
    pass


def grid(n: int) -> npt.NDArray[np.int_]:
    return np.arange(0, n, 1)


def trapezoid(N: int) -> npt.NDArray[np.float_]:
    h = np.ones(N)
    h[0] = 0.5
    return h


def simpson(N: int) -> npt.NDArray[np.float_]:
    h = np.ones(N)
    h[1::2] = 4
    h[2::2] = 2
    return h / 3


class Transform:
    """Transforms for Option pricing"""

    def __init__(
        self,
        n: int,
        max_frequency: float,
        domain_range: Bounds,
        simpson_rule: bool = False,
    ) -> None:
        self.delta_f = max_frequency / n
        self.freq = self.delta_f * grid(n)
        self.domain_range = domain_range
        self.h = simpson(n) if simpson_rule else trapezoid(n)

    @property
    def n(self) -> int:
        return self.freq.shape[0]

    @property
    def fft_zeta(self) -> float:
        return 2 * np.pi / self.n

    @property
    def fft_delta_x(self) -> float:
        return self.fft_zeta / self.delta_f

    def domain(self, delta_x: float) -> npt.NDArray[np.float_]:
        b0 = max(self.domain_range.lb, -0.5 * delta_x * self.n)
        b1 = min(self.domain_range.ub, delta_x * self.n + b0)
        if not np.isclose((b1 - b0) / self.n, delta_x):
            raise TransformError("Incompatible delta_x with domain bounds")
        return delta_x * grid(self.n) + b0

    def __call__(
        self, y: np.ndarray, delta_x: float | None = None
    ) -> dict[str, np.ndarray]:
        return self.fft(y) if delta_x is None else self.frft(y, delta_x)

    def fft(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Transform using the Fast Fourier Transform"""
        delta_x = self.fft_zeta / self.delta_f
        x, f = self.transform(y, delta_x)
        return dict(x=x, y=np.fft.fft(f).real / self.n)

    def frft(self, y: np.ndarray, delta_x: float) -> dict[str, np.ndarray]:
        x, f = self.transform(y, delta_x)
        r = frft.calculate(f, delta_x * self.delta_f)
        return dict(x=x, y=r.result.real)

    def transform(self, y: np.ndarray, delta_x: float) -> tuple[np.ndarray, np.ndarray]:
        if y.shape != self.freq.shape:
            raise TransformError("shapes not compatible")
        x = self.domain(delta_x)
        b = -x[0]
        t = self.h * self.n * np.exp(1j * self.freq * b) * y * self.delta_f / np.pi
        return x, t


@dataclass
class frft:
    """Fractional Fourier Transfrom"""

    result: np.ndarray
    zeta: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    fft_y: np.ndarray
    fft_z: np.ndarray
    y_z: np.ndarray

    @property
    def n(self) -> int:
        return self.result.shape[0]

    @property
    def fft_zeta(self) -> float:
        return 2 * np.pi / self.n

    @classmethod
    def calculate(cls, x: np.ndarray, zeta: float) -> frft:
        n = x.shape[0]
        g = grid(n)
        ez = coef(g, zeta)
        # ez2 = np.flip(ez)
        ez2 = coef(n - g, zeta)
        ezi = 1 / ez
        y = np.concatenate((x * ezi, np.zeros(n)))
        z = np.concatenate((ez, ez2))
        fft_y = np.fft.fft(y)
        fft_z = np.fft.fft(z)
        y_z = np.fft.ifft(fft_y * fft_z) / n
        result = ezi * y_z[:n]
        return cls(
            result=result,
            x=x,
            zeta=zeta,
            y=y,
            z=z,
            fft_y=fft_y,
            fft_z=fft_z,
            y_z=y_z,
        )


def coef(g: np.ndarray, zeta: float) -> np.ndarray:
    return np.exp(0.5 * 1j * g * g * zeta)
