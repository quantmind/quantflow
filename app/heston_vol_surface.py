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

    We conside a Jump Diffuxion volatility surface and compare it with a more powerful Hest Stochastioc volatility with Jumps.
    """)
    return


@app.cell
def _(mo):
    from quantflow.sp.jump_diffusion import JumpDiffusion
    from quantflow.sp.heston import HestonJ
    from quantflow.utils.distributions import DoubleExponential
    from quantflow.options.pricer import OptionPricer
    import numpy as np


    models = dict(jd="Jump Diffusion", hj="Heston with Jumps")
    model = mo.ui.dropdown({v: n for n, v in models.items()}, value="Jump Diffusion", label="model")
    vol = mo.ui.slider(start=0.1, stop=0.8, value=0.4, debounce=True, label="Long term volatility")
    sigma = mo.ui.slider(start=0.1, stop=2, step=0.1, value=0.5, debounce=True, label="vol of vol")
    kappa = mo.ui.slider(start=0.1, stop=2, step=0.5, value=0.5, debounce=True, label="Variance mean reversion")
    rho = mo.ui.slider(start=-0.6, stop=0.6, step=0.1, value=0, debounce=True, label="Correlation")
    r = mo.ui.slider(start=0.6, stop=1.6, step=0.1, value=1, debounce=True, label="Initial vol")
    jump_fraction = mo.ui.slider(start=0.1, stop=0.9, step=0.05, value=0.5, debounce=True, label="Jump Fraction")
    jump_intensity = mo.ui.slider(start=10, stop=100, step=5, debounce=True, label="Jump Intensity")
    jump_asymmetry = mo.ui.slider(start=-2, stop=2, step=0.1, value=0, debounce=True, label="Jump Asymmetry")


    class Surface:

        def __init__(self):
            self.fig = None

        @property
        def scene_camera(self):
            return self.fig.layout["scene"]["camera"] if self.fig is not None else None

        def plot(self, model, pr):
            name = models[model]
            self.fig = pr.plot3d(
                ttm=np.linspace(start=0.1, stop=1.0, num=10),
                title=f"Implied Volatility Surface: {name}",
                scene_camera=self.scene_camera
            )
            return self.fig

    surface = Surface()
    return (
        DoubleExponential,
        HestonJ,
        JumpDiffusion,
        OptionPricer,
        jump_asymmetry,
        jump_fraction,
        jump_intensity,
        kappa,
        model,
        r,
        rho,
        sigma,
        surface,
        vol,
    )


@app.cell
def _(
    DoubleExponential,
    HestonJ,
    JumpDiffusion,
    OptionPricer,
    jump_asymmetry,
    jump_fraction,
    jump_intensity,
    kappa,
    model,
    r,
    rho,
    sigma,
    vol,
):
    def get_model(value: str):
        match value:
            case "jd":
                return JumpDiffusion.create(
                    DoubleExponential,
                    vol=vol.value,
                    jump_fraction=jump_fraction.value,
                    jump_intensity=jump_intensity.value,
                    jump_asymmetry=jump_asymmetry.value,
                )
            case "hj":
                st = sigma.value/vol.value
                k = max((kappa.value, 0.5*st*st))
                return HestonJ.create(
                    DoubleExponential,
                    rate=r.value,
                    vol=vol.value,
                    sigma=sigma.value,
                    kappa=k,
                    rho=rho.value,
                    jump_fraction=jump_fraction.value,
                    jump_intensity=jump_intensity.value,
                    jump_asymmetry=jump_asymmetry.value,
                )
            case _:
                raise ValueError(f"Mode {value} not supported")

    vm = get_model(model.value)
    pricer = OptionPricer(model=vm)
    return (pricer,)


@app.cell
def _(
    jump_asymmetry,
    jump_fraction,
    jump_intensity,
    kappa,
    mo,
    model,
    r,
    rho,
    sigma,
    vol,
):
    mo.vstack([
        mo.hstack([jump_fraction, jump_intensity, jump_asymmetry]),
        mo.hstack([vol, sigma, kappa]),
        mo.hstack([rho, r, model]),
    ])
    return


@app.cell
def _(model, pricer, surface):
    fig = surface.plot(model.value, pricer)
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
