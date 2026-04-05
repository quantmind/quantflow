from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from typing_extensions import Annotated, Doc

from .network import DIVFMNetwork
from .weights import DIVFMWeights


@dataclass
class DayData:
    """Option data for a single trading day.

    Used as the unit of input for
    [DIVFMTrainer][quantflow.options.divfm.trainer.DIVFMTrainer].
    Each instance holds all options observed on one day.
    """

    moneyness_ttm: np.ndarray
    """Shape (N,). Time-scaled moneyness M = log(K/F) / sqrt(tau)."""
    ttm: np.ndarray
    """Shape (N,). Time-to-maturity tau in years."""
    implied_vols: np.ndarray
    """Shape (N,). Observed implied volatilities."""
    extra: np.ndarray | None = None
    """Shape (N, extra_features) or None. Additional observable features X."""


def _day_loss(
    network: DIVFMNetwork,
    day: DayData,
    ridge: float,
) -> torch.Tensor:
    """OLS residual loss for a single day (differentiable w.r.t. network weights).

    Computes the closed-form OLS estimate of beta_t via the normal equations
    with a small ridge penalty for numerical stability, then returns the
    squared residual norm ||IV - F @ beta||^2.
    """
    M = torch.tensor(day.moneyness_ttm, dtype=torch.float32)
    T = torch.tensor(day.ttm, dtype=torch.float32)
    IV = torch.tensor(day.implied_vols, dtype=torch.float32)
    extra = (
        torch.tensor(day.extra, dtype=torch.float32) if day.extra is not None else None
    )

    F = network(M, T, extra)  # (N, p)

    # Normal equations with ridge: beta = (F^T F + ridge * I)^{-1} F^T IV
    # torch.linalg.lstsq does not support autograd, so we use solve() instead.
    p = F.shape[1]
    FtF = F.T @ F + ridge * torch.eye(p, dtype=torch.float32)
    beta = torch.linalg.solve(FtF, F.T @ IV)  # (p,)

    residual = IV - F @ beta  # (N,)
    return (residual**2).sum()


class DIVFMTrainer:
    """Training loop for [DIVFMNetwork][quantflow.options.divfm.network.DIVFMNetwork].

    Implements the mini-batch procedure from Gauthier, Godin & Legros (2025):
    at each gradient step a random subset of days is sampled from the training
    set, OLS factor loadings are computed in closed form for each day, and the
    network weights theta are updated to minimise the total IV residual.

    The OLS step is fully differentiable via the normal equations, so gradients
    flow through beta_t back into the network parameters theta.
    """

    def __init__(
        self,
        network: Annotated[DIVFMNetwork, Doc("The network to train")],
        lr: Annotated[float, Doc("Adam learning rate")] = 1e-3,
        batch_days: Annotated[
            int,
            Doc("Number of days sampled per gradient step (J=64 in the paper)"),
        ] = 64,
        weight_decay: Annotated[float, Doc("L2 regularisation for Adam")] = 0.0,
        ridge: Annotated[
            float,
            Doc(
                "Ridge penalty added to F^T F before solving the normal equations,"
                " for numerical stability"
            ),
        ] = 1e-6,
    ) -> None:
        self.network = network
        self.batch_days = batch_days
        self.ridge = ridge
        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def step(
        self,
        days: Annotated[
            Sequence[DayData],
            Doc("Pool of training days to sample from"),
        ],
    ) -> float:
        """Perform a single gradient update step.

        Samples ``batch_days`` distinct days, computes the OLS loss for each,
        and updates the network weights.

        Returns the total loss for this step.
        """
        self.network.train()
        batch = random.sample(list(days), min(self.batch_days, len(days)))

        self.optimizer.zero_grad()
        loss: torch.Tensor = sum(  # type: ignore[assignment]
            _day_loss(self.network, day, self.ridge) for day in batch
        )
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        return loss.detach().item()

    def evaluate(
        self,
        days: Annotated[Sequence[DayData], Doc("Days to evaluate on")],
    ) -> float:
        """Compute the average per-day loss without updating weights."""
        if not days:
            return 0.0
        self.network.eval()
        total = 0.0
        with torch.no_grad():
            for day in days:
                total += float(_day_loss(self.network, day, self.ridge))
        return total / len(days)

    def fit(
        self,
        days: Annotated[Sequence[DayData], Doc("Training days")],
        num_steps: Annotated[
            int,
            Doc("Number of gradient update steps"),
        ] = 1000,
        val_days: Annotated[
            Sequence[DayData] | None,
            Doc("Optional validation days for loss monitoring"),
        ] = None,
        log_every: Annotated[
            int,
            Doc("Print a progress line every this many steps (0 to disable)"),
        ] = 100,
    ) -> list[float]:
        """Train the network for ``num_steps`` gradient steps.

        At each step, ``batch_days`` distinct days are sampled from ``days``,
        following the mini-batch procedure described in the paper.

        Returns the list of per-step training losses.
        """
        losses: list[float] = []
        for step_idx in range(1, num_steps + 1):
            loss = self.step(days)
            losses.append(loss)

            if log_every and step_idx % log_every == 0:
                msg = f"step {step_idx}/{num_steps}  loss={loss:.6f}"
                if val_days is not None:
                    val_loss = self.evaluate(val_days)
                    msg += f"  val_loss={val_loss:.6f}"
                print(msg)

        return losses

    def to_weights(self) -> DIVFMWeights:
        """Extract the trained network into a
        [DIVFMWeights][quantflow.options.divfm.weights.DIVFMWeights] instance
        ready for torch-free inference."""
        self.network.eval()
        return self.network.to_weights()
