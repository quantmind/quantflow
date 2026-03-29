import marimo

__generated_with = "0.20.4"
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
def _(mo):
    inverse = mo.ui.checkbox(value=True, label="Inverse options")
    inverse
    return (inverse,)


@app.cell
async def _(inverse):
    import pandas as pd
    from quantflow.data.deribit import Deribit

    async with Deribit() as cli:
        loader = await cli.volatility_surface_loader("eth", exclude_open_interest=0, inverse=inverse.value)

    # build the volatility surface
    surface = loader.surface()
    # calculate black implied volatilities
    surface.bs()
    # disable outliers
    surface.disable_outliers()
    # display inputs - only options with converged implied volatility
    surface_inputs = surface.inputs(converged=True)
    pd.DataFrame([i.model_dump() for i in surface_inputs.inputs])
    return (surface,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Volatility Surface
    """)
    return


@app.cell
def _(surface):
    surface.plot3d()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Term Structure
    """)
    return


@app.cell
def _(surface):
    surface.term_structure()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
