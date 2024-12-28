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

+++

## Study the Weiner process OHLC

We want to construct a mechanism to estimate the hurst exponent via OHLC data.
In order to evaluate results agains known solution we take the Weiner process as generator of timeseries. In this way we know exactly what the variance should be.

```{code-cell} ipython3
from quantflow.sp.weiner import WeinerProcess
from quantflow.utils.dates import start_of_day
p = WeinerProcess(sigma=0.5)
paths = p.sample(1, 1, 24*60*60)
paths.plot()
```

```{code-cell} ipython3
df = paths.as_datetime_df(start=start_of_day()).reset_index()
df
```

At this point we estimate the standard deviation using the **realized variance** along the path (we use the scaled flag so that the standard deviation is caled by the square-root of time step)

```{code-cell} ipython3
float(paths.path_std(scaled=True)[0])
```

```{code-cell} ipython3
from quantflow.ta.ohlc import OHLC
from dataclasses import replace
from datetime import timedelta
ohlc = OHLC(serie="0", period="10m", rogers_satchell_variance=True, parkinson_variance=True, garman_klass_variance=True)
ohlc(df)
```

```{code-cell} ipython3
import pandas as pd
results = []
for period in ("2m", "5m", "10m", "30m", "1h", "4h"):
    operator = ohlc.model_copy(update=dict(period=period))
    result = operator(df).sum()
    results.append(dict(period=period, pk=result["0_pk"].item(), gk=result["0_gk"].item(), rs=result["0_rs"].item()))
vdf = pd.DataFrame(results)
vdf
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
7*7+3*3
```

```{code-cell} ipython3

```
