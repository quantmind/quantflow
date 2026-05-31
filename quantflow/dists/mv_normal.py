import numpy as np
from pydantic import Field
from scipy.stats import multivariate_normal
from typing_extensions import Annotated

from quantflow.utils.types import FloatArray

from .base import Distribution, MeanAndCov


class MvNormal(Distribution, arbitrary_types_allowed=True):
    r"""Multivariate normal distribution $\mathcal{N}(\mu, \Sigma)$."""

    mean: Annotated[
        FloatArray,
        Field(description="Mean vector $\\mu$ of shape $(n,)$."),
    ]
    cov: Annotated[
        FloatArray,
        Field(description="Covariance matrix $\\Sigma$ of shape $(n, n)$."),
    ]

    def sample(self, size: int = 1) -> FloatArray:
        return np.asarray(
            multivariate_normal(mean=self.mean, cov=self.cov).rvs(size=size)
        )

    def log_pdf(self, x: FloatArray) -> FloatArray:
        return np.asarray(multivariate_normal(mean=self.mean, cov=self.cov).logpdf(x))

    def mean_and_cov(self) -> MeanAndCov:
        return MeanAndCov(self.mean, self.cov)
