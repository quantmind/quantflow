import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Supermsoother & EWMA
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from quantflow.data.fmp import FMP
    fmp = FMP()
    return (fmp,)


@app.cell
async def _(fmp):
    from datetime import date
    data = await fmp.prices("BTCUSD", from_date=date(2024,1,1))
    data
    return (data,)


@app.cell
def _():
    import altair as alt
    import pandas as pd
    return (alt,)


@app.cell
def _(mo):
    period = mo.ui.slider(start=2, stop=100, step=1, value=10, label="Period:")
    period
    return (period,)


@app.cell
def _(data, period):
    from quantflow.ta.supersmoother import SuperSmoother
    from quantflow.ta.ewma import EWMA
    # create the filters
    smoother = SuperSmoother(period=period.value)
    ewma = EWMA(period=period.value)
    ewma_min = EWMA(period=period.value, tau=0.5)
    # sort dates ascending
    sm = data[["date", "close"]].copy().sort_values("date", ascending=True).reset_index(drop=True)
    sm["supersmoother"] = sm["close"].apply(smoother.update)
    sm["ewma"] = sm["close"].apply(ewma.update)
    sm["ewma_min"] = sm["close"].apply(ewma_min.update)
    return (sm,)


@app.cell(hide_code=True)
def _(alt, sm):
    # Melt the dataframe to a long format suitable for Altair
    sm_long = sm.melt(
        id_vars=['date'],
        value_vars=['close', 'supersmoother', "ewma", "ewma_min"],
        var_name='Signal',
        value_name='Price'
    )

    # Create the chart with both SuperSmoothers
    line_chart_combined = alt.Chart(sm_long).mark_line().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('Price:Q', title='Price (USD)', scale=alt.Scale(zero=False)),
        color=alt.Color('Signal:N', title='Signal',
                        scale=alt.Scale(
                            domain=['close', 'supersmoother', 'ewma', "ewma_min"],
                            range=['#4c78a8', '#f58518', '#e45756', '#e45756'])  # Vega-Lite default palette
                       ),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('Signal:N', title='Signal'),
            alt.Tooltip('Price:Q', title='Price', format='$,.2f')
        ]
    ).properties(
        title='BTCUSD Close Price vs. SuperSmoothers'
    ).interactive()

    line_chart_combined
    return


@app.cell
def _(sm):
    sm
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
