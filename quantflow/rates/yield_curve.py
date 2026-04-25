from abc import ABC, abstractmethod
from decimal import Decimal

from pydantic import BaseModel


class YieldCurve(BaseModel, ABC, extra="forbid"):
    """Abstract base class for yield curves"""

    @abstractmethod
    def instanteous_forward_rate(self, ttm: float) -> Decimal:
        r"""Calculate the instantaneous forward rate for a given time to maturity

        The instantaneous forward rate is related to discount factor
        by the following formula:

        \begin{equation}
            f(\tau) = -\frac{\partial \ln D(\tau)}{\partial \tau}
        \end{equation}

        where $D(\tau)$ is the discount factor for a given time to maturity $\tau$.
        """

    @abstractmethod
    def discount_factor(self, ttm: float) -> Decimal:
        r"""Calculate the discount factor for a given time to maturity

        The discount factor is related to the instantaneous forward rate
        by the following formula:

        \begin{equation}
            D(\tau) = \exp{\left(-\int_0^\tau f(s) ds\right)}
        \end{equation}

        where $f(\tau)$ is the instantaneous forward rate for a given time to
        maturity $\tau$.
        """
