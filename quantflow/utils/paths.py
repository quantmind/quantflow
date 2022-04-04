from typing import List

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz


class Paths:
    def __init__(self, t: float, data: np.array) -> None:
        self.t = t
        self.data = data

    @property
    def dt(self) -> int:
        return self.t / self.steps

    @property
    def samples(self) -> int:
        return self.data.shape[1]

    @property
    def steps(self) -> int:
        return self.data.shape[0] - 1

    @property
    def time(self) -> np.array:
        return np.linspace(0.0, self.t, num=self.steps + 1)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    @property
    def xs(self) -> List:
        """Time as list of list (for visualization tools)"""
        return self.samples * [self.time]

    @property
    def ys(self) -> List:
        """Paths as list of list (for visualization tools)"""
        return self.data.transpose().tolist()

    def integrate(self) -> "Paths":
        """Integrate paths"""
        return self.__class__(
            self.t, cumtrapz(self.data, dx=self.dt, axis=0, initial=0)
        )
