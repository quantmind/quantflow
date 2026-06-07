# Distributions

The `dists` module collects the probability distributions used across quantflow,
both standalone parametric laws and the marginal distributions implied by a
stochastic process at a fixed time horizon.

Every distribution derives from
[Distribution][quantflow.dists.Distribution], which exposes a common
[sample][quantflow.dists.Distribution.sample] method for drawing random variates.
