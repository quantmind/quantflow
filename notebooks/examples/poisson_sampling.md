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

# Poisson Sampling

Evaluate the MC simulation for The Poisson process against the analytical PDF.

```{code-cell}
from quantflow.sp.poisson import PoissonProcess
from quantflow.utils import plot
import ipywidgets as widgets
import plotly.graph_objects as go

def simulate():
    pr = PoissonProcess(intensity=intensity.value)
    paths = pr.sample(samples.value, 1, 1000)
    pdf = paths.pdf(delta=1)
    pdf["simulation"] = pdf["pdf"]
    pdf["analytical"] = pr.marginal(1).pdf(pdf.index)
    return pdf

def on_intensity_change(change):
    df = simulate()
    fig.data[0].x = df.index
    fig.data[0].y = df["simulation"]
    fig.data[1].x = df.index
    fig.data[1].y = df["analytical"]

intensity = widgets.IntSlider(description="intensity")
samples = widgets.IntSlider(description="paths", min=100, max=10000, step=100)
intensity.value = 50
samples.value = 1000
intensity.observe(on_intensity_change)
samples.observe(on_intensity_change)

df = simulate()
simulation = go.Bar(x=df.index, y=df["simulation"], name="simulation")
analytical = go.Scatter(x=df.index, y=df["analytical"], name="analytical")
fig = go.FigureWidget(data=[simulation, analytical])

widgets.VBox([intensity, samples, fig])
```

```{code-cell}

```
