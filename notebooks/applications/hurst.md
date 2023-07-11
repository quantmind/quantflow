---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Hurst Exponent

The [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent) is used as a measure of long-term memory of time series. It relates to the autocorrelations of the time series, and the rate at which these decrease as the lag between pairs of values increases.

It is a statistics which can be used to test if a time-series is mean reverting or it is trending.

```{code-cell} ipython3
from quantflow.sp.cir import CIR

p = CIR(kappa=1, sigma=1)
```

# Links

* [Wikipedia](https://en.wikipedia.org/wiki/Hurst_exponent)
* [Hurst Exponent for Algorithmic Trading
](https://robotwealth.com/demystifying-the-hurst-exponent-part-1/)

```{code-cell} ipython3

```
