---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Calibration

Early pointers

* https://github.com/rlabbe/filterpy
* [filterpy book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

```{code-cell} ipython3
import numpy as np
np.zeros(10)
```

## Calibrating ABC

For calibration we use {cite:p}`ukf`.
Lets consider the Heston model as a test case

```{code-cell} ipython3
from quantflow.sp.heston import Heston

pr = Heston.create(vol=0.6, kappa=1.3, sigma=0.8, rho=-0.6)
pr.variance_process.is_positive
```

The Heston model is a classical example where the calibration of parameers requires to deal with the estimation of an unobserved random variable, the stochastic variance.

```{code-cell} ipython3
[p for p in pr.variance_process.parameters]
```

```{code-cell} ipython3
pr.variance_process.sample(10, t=1, steps=100)
```

```{code-cell} ipython3

```
