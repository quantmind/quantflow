---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Hurst Exponent

The [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent) is used as a measure of long-term memory of time series. It relates to the auto-correlations of the time series, and the rate at which these decrease as the lag between pairs of values increases.
It is a statistics which can be used to test if a time-series is mean reverting or it is trending.

The idea idea behind the Hurst exponent is that if the time-series $x_t$ follows a Brownian motion (aka Weiner process), than variance between two time points will increase linearly with the time difference. that is to say

\begin{align}
  \text{Var}(x_{t_2} - x_{t_1}) &\propto t_2 - t_1 \\
  &\propto \Delta t^{2H}\\
  H &= 0.5
\end{align}

where $H$ is the Hurst exponent.

Trending time-series have a Hurst exponent H > 0.5, while mean reverting time-series have H < 0.5. Understanding in which regime a time-series is can be useful for trading strategies.

These are some references to understand the Hurst exponent and its applications:

* [Hurst Exponent for Algorithmic Trading](https://robotwealth.com/demystifying-the-hurst-exponent-part-1/)
* [Basics of Statistical Mean Reversion Testing](https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/)

## Study with the Weiner Process

We want to construct a mechanism to estimate the Hurst exponent via OHLC data because it is widely available from data providers and easily constructed as an online signal during trading.

In order to evaluate results against known solutions, we consider the Weiner process as generator of timeseries.

The Weiner process is a continuous-time stochastic process named in honor of Norbert Wiener. It is often also called Brownian motion due to its historical connection with the physical model of Brownian motion of particles in water, named after the botanist Robert Brown.

We use the **WeinerProcess** from the stochastic process library and sample one path over a time horizon of 1 (day) with a time step every second.

```{code-cell} ipython3
from quantflow.sp.weiner import WeinerProcess
p = WeinerProcess(sigma=2.0)
paths = p.sample(n=1, time_horizon=1, time_steps=24*60*60)
paths.plot(title="A path of Weiner process with sigma=2.0")
```

In order to down-sample the timeseries, we need to convert it into a dataframe with dates as indices.

```{code-cell} ipython3
from quantflow.utils.dates import start_of_day
df = paths.as_datetime_df(start=start_of_day(), unit="d").reset_index()
df
```

### Realized Variance

At this point we estimate the standard deviation using the **realized variance** along the path (we use the **scaled** flag so that the standard deviation is scaled by the square-root of time step, in this way it removes the dependency on the time step size).
The value should be close to the **sigma** of the WeinerProcess defined above.

```{code-cell} ipython3
float(paths.paths_std(scaled=True)[0])
```

The evaluation of the hurst exponent is done by calculating the variance for several time windows and by fitting a line to the log-log plot of the variance vs the time window.

```{code-cell} ipython3
paths.hurst_exponent()
```

As expected, the Hurst exponent should be close to 0.5, since we have calculated the exponent from the paths of a Weiner process.

+++

### Range-based Variance Estimators

We now turn our attention to range-based variance estimators. These estimators depends on OHLC timeseries, which are widely available from data providers such as [FMP](https://site.financialmodelingprep.com/).
To analyze range-based variance estimators, we use he **quantflow.ta.OHLC** tool which allows to down-sample a timeserie to OHLC series and estimate variance with three different estimators

* **Parkinson** (1980)
* **Garman & Klass** (1980)
* **Rogers & Satchell** (1991)

See {cite:p}`molnar` for a detailed overview of the properties of range-based estimators.

For this we build an OHLC estimator as template and use it to create OHLC estimators for different periods.

```{code-cell} ipython3
import pandas as pd
import polars as pl
import math
from quantflow.ta.ohlc import OHLC
template = OHLC(serie="0", period="10m", rogers_satchell_variance=True, parkinson_variance=True, garman_klass_variance=True)
seconds_in_day = 24*60*60

def rstd(pdf: pl.Series, range_seconds: float) -> float:
    """Calculate the standard deviation from a range-based variance estimator"""
    variance = pdf.mean()
    # scale the variance by the number of seconds in the period
    variance = seconds_in_day * variance / range_seconds
    return math.sqrt(variance)

results = []
for period in ("10s", "20s", "30s", "1m", "2m", "3m", "5m", "10m", "30m"):
    ohlc = template.model_copy(update=dict(period=period))
    rf = ohlc(df)
    ts = pd.to_timedelta(period).to_pytimedelta().total_seconds()
    data = dict(period=period)
    for name in ("pk", "gk", "rs"):
        estimator = rf[f"0_{name}"]
        data[name] = rstd(estimator, ts)
    results.append(data)
vdf = pd.DataFrame(results).set_index("period")
vdf
```

These numbers are different from the realized variance because they are based on the range of the prices, not on the actual prices. The realized variance is a more direct measure of the volatility of the process, while the range-based estimators are more robust to market microstructure noise.

The Parkinson estimator is always higher than both the Garman-Klass and Rogers-Satchell estimators, the reason is due to the use of the high and low prices only, which are always further apart than the open and close prices. The GK and RS estimators are similar and are more accurate than the Parkinson estimator, especially for greater periods.

```{code-cell} ipython3
pd.options.plotting.backend = "plotly"
fig = vdf.plot(markers=True, title="Weiner Standard Deviation from Range-based estimators - correct value is 2.0")
fig.show()
```

To estimate the Hurst exponent with the range-based estimators, we calculate the variance of the log of the range for different time windows and fit a line to the log-log plot of the variance vs the time window.

```{code-cell} ipython3
from typing import Sequence
import numpy as np
from quantflow.ta.ohlc import OHLC
from collections import defaultdict
from quantflow.ta.base import DataFrame

default_periods = ("10s", "20s", "30s", "1m", "2m", "3m", "5m", "10m", "30m")

def ohlc_hurst_exponent(
    df: DataFrame,
    series: Sequence[str],
    periods: Sequence[str] = default_periods,
) -> DataFrame:
    results = {}
    estimator_names = ("pk", "gk", "rs")
    for serie in series:
        template = OHLC(
            serie=serie,
            period="10m",
            rogers_satchell_variance=True,
            parkinson_variance=True,
            garman_klass_variance=True
        )
        time_range = []
        estimators = defaultdict(list)
        for period in periods:
            ohlc = template.model_copy(update=dict(period=period))
            rf = ohlc(df)
            ts = pd.to_timedelta(period).to_pytimedelta().total_seconds()
            time_range.append(ts)
            for name in estimator_names:
                estimators[name].append(rf[f"{serie}_{name}"].mean())
        results[serie] = [float(np.polyfit(np.log(time_range), np.log(estimators[name]), 1)[0])/2.0 for name in estimator_names]
    return pd.DataFrame(results, index=estimator_names)
```

```{code-cell} ipython3
ohlc_hurst_exponent(df, series=["0"])
```

The Hurst exponent should be close to 0.5, since we have calculated the exponent from the paths of a Weiner process. But the Hurst exponent is not exactly 0.5 because the range-based estimators are not the same as the realized variance. Interestingly, the Parkinson estimator gives a Hurst exponent closer to 0.5 than the Garman-Klass and Rogers-Satchell estimators.

## Mean Reverting Time Series

We now turn our attention to mean reverting time series, where the Hurst exponent is less than 0.5.

```{code-cell} ipython3
from quantflow.sp.ou import Vasicek
import pandas as pd
pd.options.plotting.backend = "plotly"

p = Vasicek(kappa=2)
paths = {f"kappa={k}": Vasicek(kappa=k).sample(n=1, time_horizon=1, time_steps=24*60*6) for k in (1.0, 10.0, 50.0, 100.0, 500.0)}
pdf = pd.DataFrame({k: p.path(0) for k, p in paths.items()}, index=paths["kappa=1.0"].dates(start=start_of_day()))
pdf.plot()
```

We can now estimate the Hurst exponent from the realized variance. As we can see the Hurst exponent decreases as we increase the mean reversion parameter.

```{code-cell} ipython3
pd.DataFrame({k: [p.hurst_exponent()] for k, p in paths.items()})
```

And we can also estimate the Hurst exponent from the range-based estimators. As we can see the Hurst exponent decreases as we increase the mean reversion parameter along the same lines as the realized variance.

```{code-cell} ipython3
ohlc_hurst_exponent(pdf.reset_index(), list(paths), periods=("10m", "20m", "30m", "1h"))
```
