from dataclasses import dataclass

import numpy as np


def grid(N: int) -> np.ndarray:
    return np.linspace(0, N, N + 1)[:-1]


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
    def calculate(cls, x: np.ndarray, zeta: float) -> "frft":
        n = x.shape[0]
        g = grid(n)
        ez = coef(g, zeta)
        ez2 = coef(n - g, zeta)  # np.flip(ez)
        ezi = 1 / ez
        y = np.concatenate((x * ezi, np.zeros(n)))
        z = np.concatenate((ez, ez2))
        fft_y = np.fft.fft(y)
        fft_z = np.fft.fft(z)
        y_z = np.fft.ifft(fft_y * fft_z, n)
        result = ezi * y_z
        return cls(
            result=result, x=x, zeta=zeta, y=y, z=z, fft_y=fft_y, fft_z=fft_z, y_z=y_z
        )


def coef(g: np.ndarray, zeta: float) -> np.ndarray:
    return np.exp(1j * np.pi * g * g * zeta)
