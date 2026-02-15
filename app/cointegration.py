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
    # Cointegration Analysis of Cryptocurrencies
    """)
    return


@app.cell
def _(mo):
    from quantflow.data.fmp import FMP

    frequency = mo.ui.dropdown(
        options={x.value: x.value for x in FMP.freq},
        value="1min",
        label="Frequency",
    )
    frequency
    return FMP, frequency


@app.cell
async def _(FMP, frequency):
    async with FMP() as cli:
        btc = await cli.prices("BTCUSD", convert_to_date=True, frequency=frequency.value)
        eth = await cli.prices("ETHUSD", convert_to_date=True, frequency=frequency.value)
        sol = await cli.prices("SOLUSD", convert_to_date=True, frequency=frequency.value)

    btc = btc.set_index("date")
    eth = eth.set_index("date")
    sol = sol.set_index("date")
    return btc, eth, sol


@app.cell
def _(btc, eth, sol):
    # Merge the three price series on the date
    prices_3 = btc[['close']].join(eth[['close']], lsuffix='_btc', rsuffix='_eth').join(sol[['close']])
    prices_3.columns = ['btc_close', 'eth_close', 'sol_close']
    prices_3 = prices_3.dropna()
    prices_3
    return (prices_3,)


@app.cell
def _(prices_3):
    import numpy as np
    log_prices_3 = np.log(prices_3)
    return (log_prices_3,)


@app.cell
def _(log_prices_3):
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    # Perform the Johansen cointegration test
    # We choose det_order=0 for a constant term in the cointegrating relation
    # and k_ar_diff=1 for the number of lags in the VAR model.
    johansen_result = coint_johansen(log_prices_3, det_order=0, k_ar_diff=1)
    deltas = johansen_result.evec[:, 0]
    deltas

    return (deltas,)


@app.cell
def _(deltas, log_prices_3):
    residuals = log_prices_3.dot(deltas)
    residual_mean = residuals.mean()
    residuals = residuals - residual_mean
    return (residuals,)


@app.cell
def _(residuals):
    import pandas as pd
    import altair as alt


    # Create a DataFrame for plotting
    residuals_df = pd.DataFrame({
        "date": residuals.index,
        "residual": residuals.values
    })

    # Create the base chart
    base = alt.Chart(residuals_df).encode(
        x=alt.X('date:T', title='Date')
    ).properties(
        title='First Cointegration Residual of BTC, ETH, and SOL',
        width='container'
    )

    # Create the line chart for the residuals
    line = base.mark_line(color='steelblue').encode(
        y=alt.Y('residual:Q', title='Residual (Spread)'),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('residual:Q', title='Residual', format='.4f')
        ]
    )

    line.interactive()
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Why Pick the Largest Eigenvalue?

        In the Johansen cointegration test, the eigenvalues ($\lambda$) are sorted in descending order, and each one corresponds to a different potential cointegrating vector.

        **The magnitude of the eigenvalue represents the strength and stability of the corresponding cointegrating relationship.**

        1.  **Strongest Relationship:** The largest eigenvalue corresponds to the linear combination of the time series that is "most stationary." This means the resulting spread (the residuals) has the strongest tendency to revert to its mean over time.

        2.  **Statistical Significance:** The test statistics used in the Johansen test (the Trace test and the Maximum Eigenvalue test) are functions of these eigenvalues. These tests help us determine how many statistically significant cointegrating relationships exist, starting from the one associated with the largest eigenvalue.

        3.  **Practical Application:** For applications like pairs trading, we want to find the most reliable and predictable long-run equilibrium relationship. By choosing the cointegrating vector associated with the largest eigenvalue, we are selecting the portfolio of assets whose value is most likely to be mean-reverting, making it the best candidate for a statistical arbitrage strategy.

        In short, picking the largest eigenvalue is equivalent to picking the **most significant and stable cointegrating vector** found by the test.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Should You Use Log Prices?

    **Yes, using log prices is generally recommended** for cointegration analysis in finance. Here is why:

    1.  **Percentage vs. Absolute Changes:** Log-prices allow the model to work with **relative percentage changes** rather than absolute dollar amounts. This is crucial when assets trade at vastly different scales (e.g., BTC at \$60k vs. SOL at \$150).
    2.  **Variance Stabilization:** Financial time series often exhibit heteroscedasticity, meaning volatility increases as the price level rises. Log transformation helps stabilize this variance, making the data more suitable for linear statistical models.
    3.  **Linearization:** Standard cointegration tests (like Johansen or Engle-Granger) look for linear combinations. Real-world economic relationships between assets are often multiplicative (based on ratios); taking logarithms converts these into linear additive relationships.
    """)
    return


if __name__ == "__main__":
    app.run()
