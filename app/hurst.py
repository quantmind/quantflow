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
    """)
    return


if __name__ == "__main__":
    app.run()
