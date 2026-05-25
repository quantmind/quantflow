Sample from a double exponential (asymmetric Laplace) distribution and compare
the empirical PDF to both the analytical PDF and the PDF recovered from the
characteristic function.

The double exponential distribution models asymmetric jump sizes in jump-diffusion
models. The asymmetry is controlled by `log_kappa`: positive values produce
heavier right tails (upward jumps more likely), negative values heavier left tails.
At `log_kappa = 0` the distribution is symmetric.

The characteristic function inversion provides an independent check that the
analytical and numerical PDFs agree.
