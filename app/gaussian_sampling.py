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
    # Gaussian Sampling

    Here we sample the gaussian OU process for different mean reversion speed and number of paths.
    """)
    return


@app.cell
def _(mo):
    import inspect
    from quantflow.sp.ou import Vasicek
    import pandas as pd

    def simulate_vasicek(kappa: float, samples: int) -> pd.DataFrame:
        pr = Vasicek(rate=0.5, kappa=kappa)
        paths = pr.sample(samples, 1, 1000)
        pdf = paths.pdf(num_bins=50)
        pdf["simulation"] = pdf["pdf"]
        pdf["analytical"] = pr.marginal(1).pdf(pdf.index)
        return pdf

    # 1. Get the source code of your function
    # Note: The function must be defined in a previous cell or imported
    try:
        source_code = inspect.getsource(simulate_vasicek)
    except OSError:
        source_code = "# Code not available (source file not found)"

    # 2. Display it inside an accordion so it doesn't clutter the view
    mo.accordion({
        "Show Simulation Code": mo.md(f"```python\n{source_code}\n```")
    })
    return (simulate_vasicek,)


@app.cell
def _(mo):
    samples = mo.ui.slider(start=100, stop=10000, step=100, value=1000, debounce=True, full_width=True)
    kappa = mo.ui.slider(start=0.1, stop=5, step=0.1, debounce=True, full_width=True)

    def input_label(text):
        return mo.Html(f"<span style='width: 250px; display: inline-block; font-weight: 500;'>{text}</span>")

    controls = mo.vstack([
        mo.hstack([input_label("Samples:"), samples], align="center"),
        mo.hstack([input_label("Kappa (Mean reversion):"), kappa], align="center")
    ])
    controls
    return kappa, samples


@app.cell
def _(kappa, samples, simulate_vasicek):
    df = simulate_vasicek(kappa=kappa.value, samples=samples.value)
    return (df,)


@app.cell
def _(df):
    import plotly.graph_objects as go
    simulation = go.Bar(x=df.index, y=df["simulation"], name="simulation")
    analytical = go.Scatter(x=df.index, y=df["analytical"], name="analytical")
    fig = go.Figure(data=[simulation, analytical])
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
