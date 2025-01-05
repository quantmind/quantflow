---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Exponential Sampling

Here we sample the Asymmetric Laplace distribution. We will set the mean to 0 and the variance to 1 so that the distribution is fully determined by the asymmetric parameter $\kappa$.

```{code-cell} ipython3
from quantflow.utils.distributions import DoubleExponential
from quantflow.utils import bins
import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go

def simulate():
    pr = DoubleExponential.from_moments(kappa=np.exp(asym.value))
    data = pr.sample(samples.value)
    pdf = bins.pdf(data, num_bins=50, symmetric=0)
    pdf["simulation"] = pdf["pdf"]
    pdf["analytical"] = pr.pdf(pdf.index)
    cha = pr.pdf_from_characteristic()
    return pdf, cha

def on_change(change):
    df, cha = simulate()
    fig.data[0].x = df.index
    fig.data[0].y = df["simulation"]
    fig.data[1].x = df.index
    fig.data[1].y = df["analytical"]
    fig.data[2].x = cha.x
    fig.data[2].y = cha.y

asym = widgets.FloatSlider(description="asymmetry (log of k)", min=-2, max=2)
samples = widgets.IntSlider(description="paths", min=100, max=10000, step=100)
asym.value = 0
samples.value = 1000
asym.observe(on_change)
samples.observe(on_change)

df, cha = simulate()
simulation = go.Bar(x=df.index, y=df["simulation"], name="simulation")
analytical = go.Scatter(x=df.index, y=df["analytical"], name="analytical")
cha = go.Scatter(x=cha.x, y=cha.y, name="from characteristic", mode="markers")
fig = go.FigureWidget(data=[simulation, cha, analytical])

widgets.VBox([asym, samples, fig])
```

```{code-cell} ipython3

```
