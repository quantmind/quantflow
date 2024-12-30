================
Jump diffusions
================

Jump-diffusions models are a class of stochastic processes that combine a diffusion process with a jump process. The jump process is a Poisson process that generates jumps in the value of the underlying asset. The jump-diffusion model is a generalization of the Black-Scholes model that allows for the possibility of large,
discontinuous jumps in the value of the underlying asset.

The most famous jump-diffusion model is the Merton model, which was introduced by Robert Merton in 1976. The Merton model assumes that the underlying asset follows a geometric Brownian motion with jumps that are normally distributed.

.. currentmodule:: quantflow.sp.jump_diffusion

.. autoclass:: JumpDiffusion
   :members:


.. autoclass:: Merton
   :members:
