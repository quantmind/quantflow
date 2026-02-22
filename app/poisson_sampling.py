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
    # Poisson Sampling

    Evaluate the MC simulation for The Poisson process against the analytical PDF.
    """)
    return


@app.cell
def _():
    from quantflow.sp.poisson import PoissonProcess
    import pandas as pd

    def simulate_poisson(intensity: float, samples: int) -> pd.DataFrame:
        pr = PoissonProcess(intensity=intensity)
        paths = pr.sample(samples, 1, 1000)
        pdf = paths.pdf(delta=1)
        pdf["simulation"] = pdf["pdf"]
        pdf["analytical"] = pr.marginal(1).pdf(pdf.index)
        return pdf
    return (simulate_poisson,)


@app.cell
def _(mo):
    samples = mo.ui.slider(start=100, stop=10000, step=100, value=1000, debounce=True, label="Samples")
    intensity = mo.ui.slider(start=2, stop=5, step=0.1, debounce=True, label="Poisson intensity $\lambda$")

    controls = mo.hstack([samples, intensity], justify="start")
    controls
    return intensity, samples


@app.cell
def _(intensity, samples, simulate_poisson):
    df = simulate_poisson(intensity=intensity.value, samples=samples.value)
    return (df,)


@app.cell
def _(df):
    import plotly.graph_objects as go
    simulation = go.Bar(x=df.index, y=df["simulation"], name="simulation")
    analytical = go.Bar(x=df.index, y=df["analytical"], name="analytical")
    fig = go.Figure(data=[simulation, analytical])
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
