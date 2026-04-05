from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from quantflow.utils.types import FloatArray


def _sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-x))


def _apply_subnet(x: FloatArray, layers: list[LayerWeights]) -> FloatArray:
    """Numpy forward pass through a subnet (eval mode: uses running BN statistics)."""
    for layer in layers:
        x = x @ layer.weight.T + layer.bias
        if layer.apply_activation:
            x = _sigmoid(x)
        # Batch norm eval mode: normalize with running statistics
        x = (x - layer.bn_mean) / np.sqrt(layer.bn_var + layer.bn_eps)
        if layer.bn_gamma is not None and layer.bn_beta is not None:
            x = layer.bn_gamma * x + layer.bn_beta
    return x


class LayerWeights(BaseModel, arbitrary_types_allowed=True):
    """Weights for a single linear layer with batch normalization.

    Combines the linear transform, optional sigmoid activation, and
    batch normalization into one unit matching the structure of each
    block in [DIVFMNetwork][quantflow.options.divfm.network.DIVFMNetwork].
    """

    weight: FloatArray = Field(description="Linear weight matrix, shape (out, in)")
    bias: FloatArray = Field(description="Linear bias vector, shape (out,)")
    bn_mean: FloatArray = Field(description="Batch norm running mean, shape (out,)")
    bn_var: FloatArray = Field(description="Batch norm running variance, shape (out,)")
    bn_gamma: FloatArray | None = Field(
        default=None,
        description=(
            "Batch norm learnable scale (gamma), shape (out,)."
            " None for fixed (affine=False) output normalization"
        ),
    )
    bn_beta: FloatArray | None = Field(
        default=None,
        description=(
            "Batch norm learnable shift (beta), shape (out,)."
            " None for fixed (affine=False) output normalization"
        ),
    )
    bn_eps: float = Field(
        default=1e-5, description="Batch norm epsilon for numerical stability"
    )
    apply_activation: bool = Field(
        default=True,
        description=(
            "Whether to apply sigmoid activation before batch norm."
            " True for hidden layers, False for the output layer"
        ),
    )


class SubnetWeights(BaseModel):
    """Extracted weights for one sub-network (hidden layers + output layer)."""

    layers: list[LayerWeights] = Field(
        description="Ordered list of layer weights from input to output"
    )

    def forward(self, x: FloatArray) -> FloatArray:
        """Run the numpy forward pass for this subnet."""
        return _apply_subnet(x, self.layers)


class DIVFMWeights(BaseModel):
    """Extracted weights of a trained
    [DIVFMNetwork][quantflow.options.divfm.network.DIVFMNetwork].

    Implements the full network forward pass in pure numpy so that
    [DIVFMPricer][quantflow.options.divfm.pricer.DIVFMPricer] has no
    torch dependency at inference time.

    Obtain an instance from a trained network via
    [DIVFMNetwork.to_weights][quantflow.options.divfm.network.DIVFMNetwork.to_weights].
    """

    subnet_ttm: SubnetWeights = Field(
        description="Weights for the time-to-maturity sub-network (f_2)"
    )
    subnet_moneyness: SubnetWeights = Field(
        description="Weights for the moneyness sub-network (f_3)"
    )
    subnet_joint: SubnetWeights | None = Field(
        default=None,
        description=(
            "Weights for the joint (M, tau) sub-network (f_4 ... f_p)."
            " None when num_factors == 3"
        ),
    )
    num_factors: int = Field(
        description="Total number of factors p (including the constant f_1)"
    )
    extra_features: int = Field(
        default=0,
        description="Number of additional observable features X beyond (M, tau)",
    )

    def forward(
        self,
        moneyness_ttm: FloatArray,
        ttm: FloatArray,
        extra: FloatArray | None = None,
    ) -> FloatArray:
        """Compute factor values for a batch of options.

        Parameters
        ----------
        moneyness_ttm:
            Shape (N,). Time-scaled moneyness M = log(K/F) / sqrt(tau).
        ttm:
            Shape (N,). Time-to-maturity tau in years.
        extra:
            Shape (N, extra_features) or None. Additional observable features.

        Returns
        -------
        FloatArray
            Shape (N, num_factors). Factor values [f_1, f_2, ..., f_p].
        """
        N = len(moneyness_ttm)

        # f_1 = 1
        f1 = np.ones((N, 1), dtype=np.float32)

        # f_2(tau [, X])
        ttm_in: FloatArray = ttm[:, None]
        if extra is not None:
            ttm_in = np.concatenate([ttm_in, extra], axis=1)
        f2 = self.subnet_ttm.forward(ttm_in)

        # f_3(M)
        f3 = self.subnet_moneyness.forward(moneyness_ttm[:, None])

        parts: list[FloatArray] = [f1, f2, f3]

        # f_4 ... f_p (M, tau [, X])
        if self.subnet_joint is not None:
            joint_in: FloatArray = np.stack([moneyness_ttm, ttm], axis=1)
            if extra is not None:
                joint_in = np.concatenate([joint_in, extra], axis=1)
            parts.append(self.subnet_joint.forward(joint_in))

        return np.concatenate(parts, axis=1)
