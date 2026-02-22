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
    # Double Exponential Sampling

    Here we sample the Asymmetric Laplace distribution, a.k.a double exponential
    We will set the mean to 0 and the variance to 1 so that the distribution is fully determined by the asymmetric parameter $\kappa$.

    ```python
    from quantflow.utils.distributions import DoubleExponential
    ```
    """)
    return


@app.cell
def _():
    from quantflow.utils.distributions import DoubleExponential
    from quantflow.utils import bins
    import numpy as np

    def simulate_double_exponential(log_kappa: float, samples: int):
        pr = DoubleExponential.from_moments(kappa=np.exp(log_kappa))
        data = pr.sample(samples)
        pdf = bins.pdf(data, num_bins=50, symmetric=0)
        pdf["simulation"] = pdf["pdf"]
        pdf["analytical"] = pr.pdf(pdf.index)
        cha = pr.pdf_from_characteristic()
        return pdf, cha
    return (simulate_double_exponential,)


@app.cell
def _(mo):
    samples = mo.ui.slider(start=100, stop=10000, step=100, value=1000, debounce=True, label="Samples")
    log_kappa = mo.ui.slider(start=-2, stop=2, step=0.1, value=0.1, debounce=True, label="Asymmetry - $\log \kappa$")

    controls = mo.hstack([samples, log_kappa], justify="start")
    controls
    return log_kappa, samples


@app.cell
def _(log_kappa, samples, simulate_double_exponential):
    df, cha = simulate_double_exponential(log_kappa.value, samples.value)
    return cha, df


@app.cell
def _(cha, df):
    import plotly.graph_objects as go

    simulation = go.Bar(x=df.index, y=df["simulation"], name="simulation")
    analytical = go.Scatter(x=df.index, y=df["analytical"], name="analytical")
    characteristic = go.Scatter(x=cha.x, y=cha.y, name="from characteristic", mode="markers")
    fig = go.Figure(data=[simulation, characteristic, analytical])
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
