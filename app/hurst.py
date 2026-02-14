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
def _():
    from quantflow.sp.weiner import WeinerProcess
    p = WeinerProcess(sigma=2.0)
    paths = p.sample(n=1, time_horizon=1, time_steps=24*60*60)
    paths.plot(title="A path of Weiner process with sigma=2.0")
    return (paths,)


@app.cell
def _(mo):
    mo.md(r"""
    In order to down-sample the timeseries, we need to convert it into a dataframe with dates as indices.
    """)
    return


@app.cell
def _(paths):

    from quantflow.utils.dates import start_of_day
    df = paths.as_datetime_df(start=start_of_day(), unit="d").reset_index()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
