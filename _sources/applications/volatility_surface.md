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

# Volatility Surface

In this notebook we illustrate the use of the Volatility Surface tool in the library. We use [deribit](https://docs.deribit.com/) options on ETHUSD as example.

First thing, fetch the data

```{code-cell} ipython3
from quantflow.data.deribit import Deribit

async with Deribit() as cli:
    loader = await cli.volatility_surface_loader("eth")
```

Once we have loaded the data, we create the surface and display the term-structure of forwards

```{code-cell} ipython3
vs = loader.surface()
vs.maturities = vs.maturities
vs.term_structure()
```

```{code-cell} ipython3
vs.spot
```

## bs method

This method calculate the implied Black volatility from option prices. By default it uses the best option in the surface for the calculation.

The `options_df` method allows to inspect bid/ask for call options at a given cross section.
Prices of options are normalized by the Forward price, in other words they are given as base currency price, in this case BTC.

Moneyness is defined as

\begin{equation}
  k = \log{\frac{K}{F}}
\end{equation}

```{code-cell} ipython3
vs.bs()
df = vs.disable_outliers(0.95).options_df()
df
```

The plot function is enabled only if [plotly](https://plotly.com/python/) is installed

```{code-cell} ipython3
from plotly.subplots import make_subplots

# consider 6 expiries
vs6 = vs.trim(6)

titles = []
for row in range(2):
    for col in range(3):
        index = row * 3 + col
        titles.append(f"Expiry {vs6.maturities[index].maturity}")
fig = make_subplots(rows=2, cols=3, subplot_titles=titles).update_layout(height=600, title="ETH Volatility Surface")
for row in range(2):
    for col in range(3):
        index = row * 3 + col
        vs6.plot(index=index, fig=fig, showlegend=False, fig_params=dict(row=row+1, col=col+1))
fig
```

The `moneyness_ttm` is defined as

\begin{equation}
\frac{1}{\sqrt{T}} \ln{\frac{K}{F}}
\end{equation}

where $T$ is the time-to-maturity.

```{code-cell} ipython3
vs6.plot3d().update_layout(height=800, title="ETH Volatility Surface", scene_camera=dict(eye=dict(x=1, y=-2, z=1)))
```

## Model Calibration

We can now use the Vol Surface to calibrate the Heston stochastic volatility model.

```{code-cell} ipython3
from quantflow.options.calibration import HestonJCalibration, OptionPricer
from quantflow.utils.distributions import DoubleExponential
from quantflow.sp.heston import HestonJ

model = HestonJ.create(DoubleExponential, vol=0.8, sigma=1.5, kappa=0.5, rho=0.1, jump_intensity=50, jump_fraction=0.3)
pricer = OptionPricer(model=model)
cal = HestonJCalibration(pricer=pricer, vol_surface=vs6, moneyness_weight=-0)
len(cal.options)
```

```{code-cell} ipython3
cal.model.model_dump()
```

```{code-cell} ipython3
cal.fit()
```

```{code-cell} ipython3
pricer.model
```

```{code-cell} ipython3
cal.plot(index=5, max_moneyness_ttm=1)
```

## 
Serialization

It is possible to save the vol surface into a json file so it can be recreated for testing or for serialization/deserialization.

```{code-cell} ipython3
with open("../tests/volsurface.json", "w") as fp:
    fp.write(vs.inputs().model_dump_json())
```

```{code-cell} ipython3
from quantflow.options.surface import VolSurfaceInputs, surface_from_inputs
import json

with  open("../tests/volsurface.json", "r") as fp:
    inputs = VolSurfaceInputs(**json.load(fp))

vs2 = surface_from_inputs(inputs)
```
