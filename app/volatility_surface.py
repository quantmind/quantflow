import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from app.utils import nav_menu
    nav_menu()
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Volatility Surface

    In this notebook we illustrate the use of the Volatility Surface tool in the library. We use [deribit](https://docs.deribit.com/) options on ETHUSD as example.

    The library provide a [VolSurfaceLoader](api/options/vol_surface/#quantflow.options.surface.VolSurfaceLoader) for Deribit:

    ```python
    import pandas as pd
    from quantflow.data.deribit import Deribit

    async with Deribit() as cli:
        loader = await cli.volatility_surface_loader("eth", exclude_open_interest=0)

    # calibrate discount curve for the quoting asset (usd)
    loader.calibrate_curves(quote_curve=NelsonSiegel)
    # build the volatility surface
    surface = loader.surface()
    # calculate black implied volatilities
    surface.bs()
    # disable outliers
    surface.disable_outliers()
    # display inputs - only options with converged implied volatility
    surface_inputs = surface.inputs(converged=True)
    pd.DataFrame([i.model_dump() for i in surface_inputs.inputs])
    ```
    """)
    return


@app.cell
def _(mo):
    asset = mo.ui.dropdown(["btc", "eth", "sol"], value="btc", label="asset")
    inverse = mo.ui.checkbox(value=True, label="Inverse options")
    mo.hstack([asset, inverse])
    return asset, inverse


@app.cell
async def _(asset, inverse, mo):
    import pandas as pd
    from quantflow.data.deribit import Deribit
    from quantflow.rates.nelson_siegel import NelsonSiegel 

    async with Deribit() as cli:
        loader = await cli.volatility_surface_loader(
            asset.value,
            inverse=inverse.value,
            use_perp=not inverse.value
        )

    loader.calibrate_curves(quote_curve=NelsonSiegel)
    # build the volatility surface
    surface = loader.surface()
    # calculate black implied volatilities
    surface.bs()
    # disable outliers
    surface.disable_outliers()
    #
    def int_or_none(v):
        try:
            return int(v)
        except TypeError:
            return None

    maturites = [c.maturity for c in surface.maturities]
    maturity_dropdown = mo.ui.dropdown(
        options={m.strftime("%Y-%m-%d"): i for i, m in enumerate(maturites)},
        label="Maturity"
    )
    maturity_dropdown
    return int_or_none, maturity_dropdown, surface


@app.cell
def _(int_or_none, maturity_dropdown, surface):
    index = int_or_none(maturity_dropdown.value)
    surface.plot3d(index=index)
    return


@app.cell
def _(surface):
    ts = surface.term_structure()
    ts
    return (ts,)


@app.cell
def _(surface, ts):
    import math
    ttm_max = 0.1*math.ceil(10*surface.maturities[-1].ttm(surface.ref_date))
    fig = surface.quote_curve.plot(ttm_max=ttm_max)
    fig.add_scatter(x=ts["ttm"], y=ts["rate"], mode="markers", name="Cross Sections", marker=dict(size=8, color="orange"))
    fig
    return


@app.cell
def _(surface):
    surface.asset_curve.plot(ttm_max=2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
