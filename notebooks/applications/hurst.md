---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
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

## Study the Weiner process OHLC 

```{code-cell} ipython3
from quantflow.sp.weiner import WeinerProcess
p = WeinerProcess(sigma=0.5)
paths = p.sample(1, 1, 1000)
df = paths.as_datetime_df().reset_index()
df
```

```{code-cell} ipython3
from quantflow.ta.ohlc import OHLC
from datetime import timedelta
ohlc = OHLC(serie="0", period="10m", rogers_satchell_variance=True, parkinson_variance=True, garman_klass_variance=True)
result = ohlc(df)
result
```

```{code-cell} ipython3

```

# Links

* [Wikipedia](https://en.wikipedia.org/wiki/Hurst_exponent)
* [Hurst Exponent for Algorithmic Trading
](https://robotwealth.com/demystifying-the-hurst-exponent-part-1/)

```{code-cell} ipython3
import pandas as pd
v = pd.to_timedelta(0.02, unit="d")
v
```

```{code-cell} ipython3
v.to_pytimedelta()
```

```{code-cell} ipython3
from quantflow.utils.dates import utcnow
pd.date_range(start=utcnow(), periods=10, freq="0.5S")
```

```{code-cell} ipython3

```
