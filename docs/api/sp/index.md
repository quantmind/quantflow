# Stochastic Processes

This page gives an overview of all stochastic processes available in the library.

## Available processes

### Diffusion

| Process | Description |
|---|---|
| [WienerProcess][quantflow.sp.wiener.WienerProcess] | Standard Brownian motion |

### Mean-reverting (intensity)

| Process | Description |
|---|---|
| [CIR][quantflow.sp.cir.CIR] | Cox-Ingersoll-Ross square-root diffusion |
| [Vasicek][quantflow.sp.ou.Vasicek] | Gaussian Ornstein-Uhlenbeck process |
| [GammaOU][quantflow.sp.ou.GammaOU] | Non-Gaussian OU process driven by a Gamma subordinator |

### Jump processes

| Process | Description |
|---|---|
| [PoissonProcess][quantflow.sp.poisson.PoissonProcess] | Homogeneous Poisson process |
| [CompoundPoissonProcess][quantflow.sp.poisson.CompoundPoissonProcess] | Poisson process with random jump sizes |
| [DSP][quantflow.sp.dsp.DSP] | Doubly stochastic (Cox) Poisson process |

### Stochastic volatility

| Process | Description |
|---|---|
| [Heston][quantflow.sp.heston.Heston] | Classical square-root stochastic volatility model |
| [HestonJ][quantflow.sp.heston.HestonJ] | Heston model with compound Poisson jumps |
| [DoubleHeston][quantflow.sp.heston.DoubleHeston] | Two independent Heston variance processes |
| [DoubleHestonJ][quantflow.sp.heston.DoubleHestonJ] | Double Heston with compound Poisson jumps on the second component |

### Jump diffusion

| Process | Description |
|---|---|
| [JumpDiffusion][quantflow.sp.jump_diffusion.JumpDiffusion] | Diffusion with compound Poisson jumps |

## Base classes

::: quantflow.sp.base.StochasticProcess

::: quantflow.sp.base.StochasticProcess1D

::: quantflow.sp.base.IntensityProcess

::: quantflow.sp.base.StochasticProcess1DMarginal
