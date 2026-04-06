import marimo

__generated_with = "0.22.0"
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
def _():
    kwargs = dict()
    return


@app.cell
def _(mo):
    asset = mo.ui.dropdown(["btc", "eth", "sol"], value="btc", label="asset")
    inverse = mo.ui.checkbox(value=True, label="Inverse options")
    mo.hstack([asset, inverse])
    return asset, inverse


@app.cell
async def _(asset, inverse):
    import pandas as pd
    from quantflow.data.deribit import Deribit

    async with Deribit() as cli:
        loader = await cli.volatility_surface_loader(
            asset.value,
            inverse=inverse.value,
            use_perp=not inverse.value
        )

    # build the volatility surface
    surface = loader.surface()
    # calculate black implied volatilities
    surface.bs()
    # disable outliers
    surface.disable_outliers()
    surface.plot3d()
    return pd, surface


@app.cell
def _(pd, surface):
    # display inputs - only options with converged implied volatility
    surface_inputs = surface.inputs(converged=True)
    pd.DataFrame([i.model_dump() for i in surface_inputs.inputs])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
