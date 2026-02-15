import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from app.utils import nav_menu
    nav_menu()
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Hurst Exponent

    The [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent) is a statistical measure used to uncover the long-term memory of a time series. It helps determine if a financial asset is purely random, trending, or mean-reverting.

    The intuition is based on how the volatility of a time series scales with time. If a time series $x_t$ follows a standard Brownian motion (a Random Walk), the variance of the changes increases linearly with the time.

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

    ## Estimate from OHLC data

    We want to construct a mechanism to estimate the Hurst exponent via OHLC data because it is widely available from data providers and easily constructed as an online signal during trading.

    In order to evaluate results against known solutions, we consider the Weiner process as generator of timeseries.

    We use the **WeinerProcess** from the stochastic process library and sample one path over a time horizon of 1 (day) with a time step every second.
    """)
    return


@app.cell
def _(mo):
    regenerate_btn = mo.ui.run_button(label="Regenerate Path", kind="info")
    regenerate_btn
    return (regenerate_btn,)


@app.cell
def _(regenerate_btn):
    from quantflow.sp.weiner import WeinerProcess
    from quantflow.utils import plot
    from quantflow.utils.dates import start_of_day

    regenerate_btn

    weiner = WeinerProcess(sigma=2.0)
    weiner_paths = weiner.sample(n=1, time_horizon=1, time_steps=24*60*60)
    weiner_df = weiner_paths.as_datetime_df(start=start_of_day(), unit="d").reset_index()

    plot.plot_lines(
        weiner_df,
        x=weiner_df.columns[0],
        y=weiner_df.columns[1],
        title="Weiner Process Path",
        labels={"value": "Value", "variable": "Path", weiner_df.columns[0]: "Date"},
    )
    return plot, start_of_day, weiner_df, weiner_paths


@app.cell
def _(mo):
    mo.md(r"""
    In order to down-sample the timeseries, we need to convert it into a dataframe with dates as indices.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Realized Variance

    At this point we estimate the standard deviation using the **realized variance** along the path (we use the **scaled** flag so that the standard deviation is scaled by the square-root of time step, in this way it removes the dependency on the time step size).
    The value should be close to the **sigma** of the WeinerProcess defined above.
    """)
    return


@app.cell
def _(weiner_paths):
    float(weiner_paths.paths_std(scaled=True)[0])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The evaluation of the hurst exponent is done by calculating the variance for several time windows and by fitting a line to the log-log plot of the variance vs the time window.
    """)
    return


@app.cell
def _(weiner_paths):
    weiner_paths.hurst_exponent()
    return


@app.cell
def _(mo):
    mo.md(r"""
    As expected, the Hurst exponent should be close to 0.5, since we have calculated the exponent from the paths of a Weiner process.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Range-based Variance Estimators

    We now turn our attention to range-based variance estimators. These estimators depends on OHLC timeseries, which are widely available from data providers such as [FMP](https://site.financialmodelingprep.com/).
    To analyze range-based variance estimators, we use he **quantflow.ta.OHLC** tool which allows to down-sample a timeserie to OHLC series and estimate variance with three different estimators

    * **Parkinson** (1980)
    * **Garman & Klass** (1980)
    * **Rogers & Satchell** (1991)

    See [molnar](/bibliography/#molnar) for a detailed overview of the properties of range-based estimators.

    For this we build an OHLC estimator as template and use it to create OHLC estimators for different periods.
    """)
    return


@app.cell
def _(weiner_df):
    import pandas as pd
    import polars as pl
    import math
    from quantflow.ta.ohlc import OHLC
    template = OHLC(
        serie="0",
        period="10m",
        rogers_satchell_variance=True,
        parkinson_variance=True,
        garman_klass_variance=True
    )
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
        rf = ohlc(weiner_df)
        ts = pd.to_timedelta(period).to_pytimedelta().total_seconds()
        data = dict(period=period)
        for name in ("pk", "gk", "rs"):
            estimator = rf[f"0_{name}"]
            data[name] = rstd(estimator, ts)
        results.append(data)
    vdf = pd.DataFrame(results).set_index("period")
    return OHLC, pd, vdf


@app.cell
def _(plot, vdf):
    # Create a scatter plot comparing the three estimators
    # We use the dataframe index (period) for the x-axis and the columns for the y-axis
    fig2 = plot.plot_scatter(
        vdf,
        x=vdf.index,
        y=vdf.columns,
        title="Range-based Volatility Estimators vs Sampling Period",
        labels={
            "period": "Sampling Period",
            "value": "Estimated Volatility (Annualized)",
            "variable": "Estimator"
        },
    )

    # Increase marker size and add opacity for better visibility
    fig2.update_traces(marker=dict(size=10, opacity=0.8))

    # Add hover lines to easily compare values at the same period
    fig2.update_layout(hovermode="x unified")

    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    These numbers are different from the realized variance because they are based on the range of the prices, not on the actual prices. The realized variance is a more direct measure of the volatility of the process, while the range-based estimators are more robust to market microstructure noise.

    The Parkinson estimator is always higher than both the Garman-Klass and Rogers-Satchell estimators, the reason is due to the use of the high and low prices only, which are always further apart than the open and close prices. The GK and RS estimators are similar and are more accurate than the Parkinson estimator, especially for greater periods.

    To estimate the Hurst exponent with the range-based estimators, we calculate the variance of the log of the range for different time windows and fit a line to the log-log plot of the variance vs the time window.
    """)
    return


@app.cell
def _(OHLC, pd):
    from typing import Sequence
    import numpy as np
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
    return (ohlc_hurst_exponent,)


@app.cell
def _(ohlc_hurst_exponent, weiner_df):
    ohlc_hurst_exponent(weiner_df, series=["0"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The Hurst exponent should be close to 0.5, since we have calculated the exponent from the paths of a Weiner process. But it is not exactly 0.5 because the range-based estimators are not the same as the realized variance. Interestingly, the Parkinson estimator gives a Hurst exponent closer to 0.5 than the Garman-Klass and Rogers-Satchell estimators.

    ## Mean Reverting Time Series

    We now turn our attention to mean reverting time series, where the Hurst exponent is less than 0.5.
    """)
    return


@app.cell
def _(pd, regenerate_vasicek, start_of_day):
    from quantflow.sp.ou import Vasicek
    regenerate_vasicek
    p = Vasicek(kappa=2)
    paths = {f"kappa={k}": Vasicek(kappa=float(k)).sample(n=1, time_horizon=1, time_steps=24*60*6) for k in (1, 10, 50, 100, 500)}
    pdf = pd.DataFrame({k: p.path(0) for k, p in paths.items()}, index=paths["kappa=1"].dates(start=start_of_day()))
    pdf.plot()
    return paths, pdf


@app.cell
def _(mo):
    regenerate_vasicek = mo.ui.run_button(label="Regenerate Paths", kind="info")
    regenerate_vasicek
    return (regenerate_vasicek,)


@app.cell
def _(mo):
    mo.md(r"""
    ~We can now estimate the Hurst exponent from the realized variance. As we can see the Hurst exponent decreases as we increase the mean reversion parameter.
    """)
    return


@app.cell
def _(paths, pd):
    pd.DataFrame({k: [p.hurst_exponent()] for k, p in paths.items()})
    return


@app.cell
def _(mo):
    mo.md(r"""
    And we can also estimate the Hurst exponent from the range-based estimators. As we can see the Hurst exponent decreases as we increase the mean reversion parameter along the same lines as the realized variance.
    """)
    return


@app.cell
def _(ohlc_hurst_exponent, paths, pdf):
    ohlc_hurst_exponent(pdf.reset_index(), list(paths), periods=("10m", "20m", "30m", "1h"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
