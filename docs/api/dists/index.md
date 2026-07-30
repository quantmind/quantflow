# Distributions

The `dists` module collects the probability distributions used across quantflow,
both standalone parametric laws and the marginal distributions implied by a
stochastic process at a fixed time horizon.

Every distribution derives from
[Distribution][quantflow.dists.Distribution], which exposes a common
[sample][quantflow.dists.Distribution.sample] method for drawing random variates.

The [1D Distributions](distributions1d.md) page documents
[Distribution1D][quantflow.dists.Distribution1D] and its concrete laws
[Normal][quantflow.dists.Normal], [Exponential][quantflow.dists.Exponential]
and [DoubleExponential][quantflow.dists.DoubleExponential], used as jump size
distributions in compound Poisson processes.

Multivariate laws derive from [MvDistribution][quantflow.dists.MvDistribution],
which exposes its [MeanAndCov][quantflow.dists.MeanAndCov] statistics;
[MvNormal][quantflow.dists.MvNormal] is the multivariate normal implementation,
documented on the [Distributions](distributions.md) page.

[Marginal1D][quantflow.dists.Marginal1D], on the [Marginal 1D](marginal1d.md)
page, is the abstract 1D distribution with Fourier based option pricing.
The pricing method is selected via
[OptionPricingMethod][quantflow.dists.OptionPricingMethod] and results are
returned as [OptionPricingResult][quantflow.dists.OptionPricingResult] or
[OptionPricingCosResult][quantflow.dists.OptionPricingCosResult].
