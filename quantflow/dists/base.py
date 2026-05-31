from abc import ABC, abstractmethod
from typing import NamedTuple

from pydantic import BaseModel
from typing_extensions import Annotated, Doc

from quantflow.utils.types import FloatArray


class MeanAndCov(NamedTuple):
    """Mean vector and covariance matrix of a distribution."""

    mean: FloatArray
    cov: FloatArray


class Distribution(BaseModel, ABC):
    """Base class for distributions."""

    @abstractmethod
    def sample(
        self,
        size: Annotated[int, Doc("Number of samples to draw.")] = 1,
    ) -> FloatArray:
        """Draw random samples from the distribution."""
        ...

    @abstractmethod
    def log_pdf(
        self,
        x: Annotated[FloatArray, Doc("Point at which to evaluate the log-density.")],
    ) -> FloatArray:
        """Log probability density at $x$."""
        ...

    @abstractmethod
    def mean_and_cov(self) -> MeanAndCov:
        """Mean vector and covariance matrix of the distribution."""
        ...
