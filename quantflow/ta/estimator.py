from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Doc

from quantflow.utils.types import FloatArray


class EstimatedResult(BaseModel):
    """Result of a time series model estimation."""

    params: dict[str, float] = Field(description="Estimated model parameters")
    objective: float = Field(
        description="Objective value at optimum (log-likelihood or moment distance)"
    )
    success: bool = Field(description="Whether the optimization converged")
    message: str = Field(description="Convergence message from the optimizer")
    std_errors: dict[str, float] | None = Field(
        default=None,
        description="Standard errors of the estimated parameters, if available",
    )


class TimeSeriesEstimator(BaseModel, ABC):
    """Base class for time series model parameter estimation."""

    @abstractmethod
    def fit(
        self,
        data: Annotated[FloatArray, Doc("Observed time series")],
        dt: Annotated[float, Doc("Time step between observations")],
    ) -> EstimatedResult:
        """Fit the model to the observed time series and return estimated parameters."""
