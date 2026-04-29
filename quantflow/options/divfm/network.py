from __future__ import annotations

import torch
import torch.nn as nn
from typing_extensions import Annotated, Doc

from .weights import DIVFMWeights, LayerWeights, SubnetWeights


def _make_subnet(
    input_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    output_size: int,
) -> nn.Sequential:
    """Feedforward subnet: affine + sigmoid + batch norm per hidden layer,
    then a linear output layer with fixed (non-trainable) batch normalization."""
    layers: list[nn.Module] = []
    in_size = input_size
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(nn.Sigmoid())
        layers.append(nn.BatchNorm1d(hidden_size))
        in_size = hidden_size
    layers.append(nn.Linear(in_size, output_size))
    # Fixed output normalization: zero mean, unit variance, no learnable scale/shift
    layers.append(nn.BatchNorm1d(output_size, affine=False))
    return nn.Sequential(*layers)


def _extract_subnet(subnet: nn.Sequential) -> SubnetWeights:
    """Extract weights from a torch subnet into a
    [SubnetWeights][quantflow.options.divfm.weights.SubnetWeights] instance."""
    modules = list(subnet.children())
    layers = []
    i = 0
    while i < len(modules):
        linear = modules[i]
        assert isinstance(linear, nn.Linear)
        W = linear.weight.detach().cpu().numpy().copy()
        b = linear.bias.detach().cpu().numpy().copy()
        i += 1
        apply_activation = i < len(modules) and isinstance(modules[i], nn.Sigmoid)
        if apply_activation:
            i += 1
        bn = modules[i]
        assert isinstance(bn, nn.BatchNorm1d)
        layers.append(
            LayerWeights(
                weight=W,
                bias=b,
                bn_mean=bn.running_mean.cpu().numpy().copy(),  # type: ignore[union-attr]
                bn_var=bn.running_var.cpu().numpy().copy(),  # type: ignore[union-attr]
                bn_gamma=(
                    bn.weight.detach().cpu().numpy().copy() if bn.affine else None
                ),
                bn_beta=(bn.bias.detach().cpu().numpy().copy() if bn.affine else None),
                bn_eps=bn.eps,
                apply_activation=apply_activation,
            )
        )
        i += 1
    return SubnetWeights(layers=layers)


class DIVFMNetwork(nn.Module):
    r"""Neural network implementing the latent factor functions

    $$
    f_p\left(m, \tau, X; \theta\right)
    $$

    Produces $P$ factor functions with the following structural constraints
    (as in [gauthier](../../bibliography.md#gauthier)):

    - $f_1 = 1$ constant, not learned
    - $f_2(\tau, X)$ depends only on time-to-maturity and optional extra features X
    - $f_3(m)$ depends only on time-scaled moneyness
    - $f_4, ..., f_p (m, \tau, X)$ unrestricted

    These structural constraints improve interpretability by associating each
    factor with a specific dimension of the implied volatility surface.

    The network uses sigmoid activations throughout to ensure the implied
    volatility surface is twice continuously differentiable in the strike
    dimension, which is required for a well-defined risk-neutral density.
    """

    def __init__(
        self,
        num_factors: Annotated[
            int,
            Doc(
                "Total number of factors p (including the constant $f_1$). "
                "Must be greater or equal 3 to satisfy the structural constraints"
            ),
        ] = 5,
        hidden_size: Annotated[
            int,
            Doc("Number of neurons per hidden layer"),
        ] = 32,
        num_hidden_layers: Annotated[
            int,
            Doc("Number of hidden layers L - 2 (default 3 gives L=5 total)"),
        ] = 3,
        extra_features: Annotated[
            int,
            Doc(
                "Number of additional observable features X beyond (M, tau),"
                " e.g. time-to-earnings-announcement"
            ),
        ] = 0,
    ) -> None:
        super().__init__()
        if num_factors < 3:
            raise ValueError(
                "num_factors must be at least 3 (constant + ttm + moneyness)"
            )
        self.num_factors = num_factors
        self.extra_features = extra_features

        # f_2: input is tau (+ optional extra features X)
        self.subnet_ttm = _make_subnet(
            input_size=1 + extra_features,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            output_size=1,
        )

        # f_3: input is M only (moneyness, no extra features by design)
        self.subnet_moneyness = _make_subnet(
            input_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            output_size=1,
        )

        # f_4 ... f_p: input is (M, tau) + optional extra features X
        num_joint = num_factors - 3
        if num_joint > 0:
            self.subnet_joint: nn.Module = _make_subnet(
                input_size=2 + extra_features,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                output_size=num_joint,
            )
        else:
            self.subnet_joint = nn.Identity()

    def to_weights(self) -> DIVFMWeights:
        """Extract network weights into a
        [DIVFMWeights][quantflow.options.divfm.weights.DIVFMWeights] instance
        for torch-free inference."""
        return DIVFMWeights(
            subnet_ttm=_extract_subnet(self.subnet_ttm),
            subnet_moneyness=_extract_subnet(self.subnet_moneyness),
            subnet_joint=(
                _extract_subnet(self.subnet_joint)  # type: ignore[arg-type]
                if self.num_factors > 3
                else None
            ),
            num_factors=self.num_factors,
            extra_features=self.extra_features,
        )

    def forward(
        self,
        moneyness_ttm: Annotated[
            torch.Tensor,
            Doc("Shape (N,). Time-scaled moneyness M = log(K/F) / sqrt(tau)"),
        ],
        ttm: Annotated[
            torch.Tensor,
            Doc("Shape (N,). Time-to-maturity tau in years"),
        ],
        extra: Annotated[
            torch.Tensor | None,
            Doc("Shape (N, extra_features) or None. Additional observable features X"),
        ] = None,
    ) -> torch.Tensor:
        """Compute factor values for a batch of options.

        Returns shape (N, num_factors) with factor values [f_1, f_2, ..., f_p].
        """
        N = moneyness_ttm.shape[0]

        # f_1 = 1
        f1 = torch.ones(N, 1, device=moneyness_ttm.device, dtype=moneyness_ttm.dtype)

        # f_2(tau [, X])
        ttm_input = ttm.unsqueeze(1)
        if extra is not None:
            ttm_input = torch.cat([ttm_input, extra], dim=1)
        f2 = self.subnet_ttm(ttm_input)

        # f_3(M)
        f3 = self.subnet_moneyness(moneyness_ttm.unsqueeze(1))

        parts = [f1, f2, f3]

        # f_4 ... f_p (M, tau [, X])
        if self.num_factors > 3:
            joint_input = torch.cat(
                [moneyness_ttm.unsqueeze(1), ttm.unsqueeze(1)]
                + ([extra] if extra is not None else []),
                dim=1,
            )
            parts.append(self.subnet_joint(joint_input))  # type: ignore[arg-type]

        return torch.cat(parts, dim=1)
