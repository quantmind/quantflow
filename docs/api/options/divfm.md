# Deep IV Factor Model

The DIVFM module implements the Deep Implied Volatility Factor Model from
Gauthier, Godin & Legros (2025). The IV surface on a given day is modelled as
a linear combination of $p$ fixed latent functions learned by a neural network:

$$\sigma_t(M, \tau; \theta) = \mathbf{f}(M, \tau, X; \theta)\,\boldsymbol{\beta}_t = \sum_{i=1}^{p} \beta_{t,i}\,f_i(M, \tau, X; \theta)$$

where $M = \frac{1}{\sqrt{\tau}}\log\!\left(\frac{K}{F_{t,\tau}}\right)$ is the
time-scaled moneyness, $\mathbf{f}$ is a feedforward neural network with fixed
weights $\theta$ shared across all days, and $\boldsymbol{\beta}_t$ are daily
coefficients fitted in closed form via OLS.

## Inference (no torch required)

::: quantflow.options.divfm.DIVFMPricer

::: quantflow.options.divfm.DIVFMWeights

::: quantflow.options.divfm.weights.SubnetWeights

::: quantflow.options.divfm.weights.LayerWeights

## Training (requires `quantflow[ml]`)

::: quantflow.options.divfm.network.DIVFMNetwork

::: quantflow.options.divfm.trainer.DIVFMTrainer

::: quantflow.options.divfm.trainer.DayData
