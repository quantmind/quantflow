---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
---

# %% [markdown]
# # Hurst Exponent
#
# The [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent) is used as a measure of long-term memory of time series. It relates to the autocorrelations of the time series, and the rate at which these decrease as the lag between pairs of values increases.
#
# It is a statistics which can be used to test if a time-series is mean reverting or it is trending.

# %% [markdown]
# ## Study with the Weiner Process
#
# We want to construct a mechanism to estimate the Hurst exponent via OHLC data because it is widely available from data provider and easily constructed as an online signal during trading.
#
# In order to evaluate results against known solutions, we consider the Weiner process as generator of timeseries.
#
# The Weiner process is a continuous-time stochastic process named in honor of Norbert Wiener. It is often also called Brownian motion due to its historical connection with the physical model of Brownian motion of particles in water, named after the botanist Robert Brown.

# %%
from quantflow.sp.weiner import WeinerProcess
from quantflow.utils.dates import start_of_day
p = WeinerProcess(sigma=0.5)
paths = p.sample(1, 1, 24*60*60)
paths.plot()

# %%
df = paths.as_datetime_df(start=start_of_day()).reset_index()
df

# %% [markdown]
# ### Realized Variance
#
# At this point we estimate the standard deviation using the **realized variance** along the path (we use the **scaled** flag so that the standard deviation is scaled by the square-root of time step, in this way it removes the dependency on the time step size).
# The value should be close to the **sigma** of the WeinerProcess defined above.

# %%
float(paths.path_std(scaled=True)[0])

# %% [markdown]
# ### Range-base Variance estimators
#
# We now turn our attention to range-based volatility estimators. These estimators depends on OHLC timeseries, which are widely available from data providers such as [FMP](https://site.financialmodelingprep.com/).
# To analyze range-based variance estimators, we use he **quantflow.ta.OHLC** tool which allows to down-sample a timeserie to OHLC series and estimate variance with three different estimators
#
# * **Parkinson** (1980)
# * **Garman & Klass** (1980)
# * **Rogers & Satchell** (1991)
#
# See {cite:p}`molnar` for a detailed overview of the properties of range-based estimators.
#
# For this we build an OHLC estimator as template and use it to create OHLC estimators for different periods.

# %%
import pandas as pd
from quantflow.ta.ohlc import OHLC
ohlc = OHLC(serie="0", period="10m", rogers_satchell_variance=True, parkinson_variance=True, garman_klass_variance=True)

results = []
for period in ("2m", "5m", "10m", "30m", "1h", "4h"):
    operator = ohlc.model_copy(update=dict(period=period))
    result = operator(df).sum()
    results.append(dict(period=period, pk=result["0_pk"].item(), gk=result["0_gk"].item(), rs=result["0_rs"].item()))
vdf = pd.DataFrame(results)
vdf

# %% [markdown]
# # Links
#
# * [Wikipedia](https://en.wikipedia.org/wiki/Hurst_exponent)
# * [Hurst Exponent for Algorithmic Trading
# ](https://robotwealth.com/demystifying-the-hurst-exponent-part-1/)

# %%
import pandas as pd
v = pd.to_timedelta(0.02, unit="d")
v

# %%
v.to_pytimedelta()

# %%
from quantflow.utils.dates import utcnow
pd.date_range(start=utcnow(), periods=10, freq="0.5S")

# %%
7*7+3*3

# %%
