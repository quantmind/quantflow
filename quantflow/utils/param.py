from dataclasses import dataclass, field
from typing import Collection, Generator, Sequence

import numpy as np
from scipy.optimize import Bounds


def default_bounds() -> Bounds:
    return Bounds(-np.inf, np.inf)


@dataclass
class Param:
    name: str
    value: float = 0
    bounds: Bounds = field(default_factory=default_bounds)
    description: str = ""


class Parameters(Collection[Param]):
    def __init__(self, *params: Param) -> None:
        self.param_dict = {}
        self.extend(params)

    def extend(self, params: Sequence[Param]) -> None:
        for p in params:
            self.append(p)

    def append(self, p: Param) -> None:
        self.param_dict[p.name] = p

    def info(self, sep: str = "\n", rd: int = 4) -> str:
        return sep.join((f"{p.name}: {round(p.value, rd)}" for p in self))

    def __len__(self) -> int:
        return len(self.param_dict)

    def __contains__(self, name: str) -> bool:
        return name in self.param_dict

    def __iter__(self) -> Generator[Param, None, None]:
        yield from self.param_dict.values()

    def values(self) -> Generator[float, None, None]:
        for p in self:
            yield p.value

    def set_values(self, x: Sequence) -> None:
        for p, v in zip(self, x):
            p.value = v

    def __repr__(self) -> str:
        return self.param_dict.__repr__()

    def __getitem__(self, name: str) -> Param:
        return self.param_dict[name]

    def __getattr__(self, name: str) -> float:
        try:
            return self.param_dict[name].value
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None
