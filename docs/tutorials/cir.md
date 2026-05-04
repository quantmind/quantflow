# CIR Process

This tutorial shows how to use the
[CIR][quantflow.sp.cir.CIR] (Cox-Ingersoll-Ross) model and validates
the analytical [marginal PDF][quantflow.sp.cir.CIR.analytical_pdf] against
the PDF recovered from the [characteristic function][quantflow.sp.cir.CIR.characteristic_exponent].

## The model

The CIR process is a mean-reverting square-root diffusion with three parameters:

| Parameter | Description |
|---|---|
| `kappa` | Mean-reversion speed |
| `theta` | Long-run mean |
| `sigma` | Volatility of volatility |
| `rate` | Initial value $x_0$ |

```python
from quantflow.sp.cir import CIR

cir = CIR(kappa=2.0, theta=0.5, sigma=0.8, rate=1.0)
print(cir.feller_condition)   # positive: process stays strictly positive
print(cir.is_positive)
```

The process stays strictly positive when the Feller condition holds:

\begin{equation}
    2\kappa\theta \geq \sigma^2
\end{equation}

## Analytical moments

The marginal distribution at time $t$ has closed-form mean and variance,
accessible via the [marginal][quantflow.sp.cir.CIR.marginal]:

```python
m = cir.marginal(1.0)
print(m.mean())       # analytical mean
print(m.variance())   # analytical variance
```

## PDF comparison

The marginal PDF has two independent routes to the same result:

* **Analytical**: the [scaled non-central chi-squared][quantflow.sp.cir.CIR.analytical_pdf]
  transition density in closed form.
* **Characteristic function**: numerical inversion of $\Phi = e^{-\phi}$ via
  [pdf_from_characteristic][quantflow.utils.marginal.Marginal1D.pdf_from_characteristic].

The charts below overlay both for a CIR process with
$\kappa=1$, $\theta=0.5$, $\sigma=0.8$, $x_0=3$, starting well above the long-run mean
to make the mean-reversion clearly visible across time horizons.

### Short horizon

At $t = 0.5$ the distribution is still centred near the initial value $x_0$:

[![CIR PDF t=0.5](../assets/examples/cir_pdf_t05.png)](../assets/examples/cir_pdf_t05.png){target="_blank"}

### Long horizon

At $t = 2.0$ the distribution has mean-reverted toward $\theta = 0.5$ and the
inversion shows visible oscillations:

[![CIR PDF t=2.0](../assets/examples/cir_pdf_t20.png)](../assets/examples/cir_pdf_t20.png){target="_blank"}

The oscillations are a Gibbs phenomenon. The CIR density has a cusp at the
origin: near $x = 0$ it grows as $x^{q/2}$ where $q = 2\kappa\theta/\sigma^2 - 1$.
When $q < 1$ the characteristic function decays algebraically as $u^{-(1+q/2)}$
rather than exponentially. For these parameters $q \approx 0.56$, so the
integral is still non-negligible when it gets truncated.

At $t = 0.5$ the mean is nearly three standard deviations from zero, so the cusp
is invisible and the inversion is accurate. By $t = 2$ the process has drifted
to within 1.4 standard deviations of the origin and the cusp affects the result.

For CIR with $q < 1$ the analytical PDF is the right tool. The inversion is
confirmed by the characteristic function plot below.

## Characteristic function

The plot below shows $|\Phi(u)|$ and $\text{Re}[\Phi(u)]$ at $t=2$. The
magnitude is still around $0.05$ at the truncation point, confirming that the
integral is cut off before it decays to zero:

[![CIR characteristic function t=2.0](../assets/examples/cir_cf_t20.png)](../assets/examples/cir_cf_t20.png){target="_blank"}

## Code

```python
--8<-- "docs/examples/cir_pdf_comparison.py"
```
