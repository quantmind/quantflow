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
    ## Jump Diffusion
    """)
    return


@app.cell
def _(mo):
    from quantflow.sp.jump_diffusion import JumpDiffusion
    from quantflow.utils.distributions import DoubleExponential

    jump_fraction = mo.ui.slider(start=0.1, stop=0.9, step=0.05, value=0.5, debounce=True, label="Jump Fraction")
    jump_intensity = mo.ui.slider(start=10, stop=100, step=5, debounce=True, label="Jump Intensity")
    jump_asymmetry = mo.ui.slider(start=-2, stop=2, step=0.1, value=0, debounce=True, label="Jump Asymmetry")
    jump_controls = mo.vstack([
        jump_fraction, jump_intensity, jump_asymmetry
    ])
    jump_controls
    return (
        DoubleExponential,
        JumpDiffusion,
        jump_asymmetry,
        jump_fraction,
        jump_intensity,
    )


@app.cell
def _(
    DoubleExponential,
    JumpDiffusion,
    OptionPricer,
    jump_asymmetry,
    jump_fraction,
    jump_intensity,
):
    jd = JumpDiffusion.create(
        DoubleExponential,
        jump_fraction=jump_fraction.value,
        jump_intensity=jump_intensity.value,
        jump_asymmetry=jump_asymmetry.value,
    )
    OptionPricer(model=jd).maturity(0.01).plot()
    return (jd,)


@app.cell
def _(jd):
    jd.model_dump()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Heston Volatility Surface
    """)
    return


@app.cell
def _(mo):
    from quantflow.options.pricer import OptionPricer
    from quantflow.sp.heston import HestonJ

    sigma = mo.ui.slider(start=0.1, stop=2, step=0.1, debounce=True, label="vol of vol")
    sigma
    return HestonJ, OptionPricer, sigma


@app.cell
def _(DoubleExponential, HestonJ, jump_asymmetry, jump_fraction, sigma):
    hj = HestonJ.create(
        DoubleExponential,
        vol=0.5,
        sigma=sigma.value,
        kappa=1.0,
        rho=0.0,
        jump_fraction=jump_fraction.value,
        jump_asymmetry=jump_asymmetry.value,
    )
    return (hj,)


@app.cell
def _(OptionPricer, hj):
    hjp = OptionPricer(model=hj)
    hjp.maturity(0.5).plot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
