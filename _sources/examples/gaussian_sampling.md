---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Gaussian Sampling

Here we sample the gaussian OU process for different mean reversion speed and number of paths.

```{admonition} Interactive notebook not enabled in docs - how to run it interactively?
The widget below is not enabled in the documentation. You can run the notebook to see the widget in action, see [contributing](../reference/contributing.md) for instructions on how to run the notebook.
```

```{code-cell} ipython3
from quantflow.sp.ou import Vasicek
from quantflow.utils import plot
import ipywidgets as widgets
import plotly.graph_objects as go

def simulate():
    pr = Vasicek(rate=0.5, kappa=kappa.value)
    paths = pr.sample(samples.value, 1, 1000)
    pdf = paths.pdf(num_bins=50)
    pdf["simulation"] = pdf["pdf"]
    pdf["analytical"] = pr.marginal(1).pdf(pdf.index)
    return pdf

def on_intensity_change(change):
    df = simulate()
    fig.data[0].x = df.index
    fig.data[0].y = df["simulation"]
    fig.data[1].x = df.index
    fig.data[1].y = df["analytical"]

kappa = widgets.FloatSlider(description="mean reversion", min=0.1, max=5)
samples = widgets.IntSlider(description="paths", min=100, max=10000, step=100)
kappa.value = 1
samples.value = 1000
kappa.observe(on_intensity_change)
samples.observe(on_intensity_change)

df = simulate()
simulation = go.Bar(x=df.index, y=df["simulation"], name="simulation")
analytical = go.Scatter(x=df.index, y=df["analytical"], name="analytical")
fig = go.FigureWidget(data=[simulation, analytical])

widgets.VBox([kappa, samples, fig])
```

```{code-cell} ipython3

```
