from dataclasses import dataclass, field
from typing import Generic, TypeVar

from quantflow.sp.base import StochasticProcess

from .surface import OptionPrice, S, VolSurface

M = TypeVar("M", bound=StochasticProcess)


@dataclass
class OptionEntry:
    """Entry for a single option"""

    options: list[OptionPrice] = field(default_factory=list)


@dataclass
class VolModelCalibration(Generic[M, S]):
    """Calibration of a stochastic volatility model"""

    model: M
    surface: VolSurface[S]
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.surface.bs()
        for option in self.surface.option_prices():
            key = (option.maturity, option.strike)
            if key not in self.options:
                entry = OptionEntry()
                self.options[key] = entry
            entry.options.append(option)
