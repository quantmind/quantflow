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

# Volatility Surface

In this notebook we illustrate the use of the Volatility Surface tool in the library. We use [deribit](https://docs.deribit.com/) options on BTCUSD as example.

First thing, fetch the data

```{code-cell} ipython3
from quantflow.data.deribit import Deribit

async with Deribit() as cli:
    loader = await cli.volatility_surface_loader("eth")
```

Once we have loaded the data, we create the surface and display the term-structure of forwards

```{code-cell} ipython3
vs = loader.surface()
vs.maturities = vs.maturities[1:]
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
vs.plot().update_layout(height=500, title="BTC Volatility Surface")
```

The `moneyness_ttm` is defined as

\begin{equation}
\frac{1}{\sqrt{T}} \ln{\frac{K}{F}}
\end{equation}

where $T$ is the time-to-maturity.

```{code-cell} ipython3
vs.plot3d().update_layout(height=800, title="BTC Volatility Surface", scene_camera=dict(eye=dict(x=1, y=-2, z=1)))
```

## Model Calibration

We can now use the Vol Surface to calibrate the Heston stochastic volatility model.

```{code-cell} ipython3
from quantflow.options.calibration import HestonCalibration, OptionPricer
from quantflow.sp.heston import Heston

pricer = OptionPricer(Heston.create(vol=0.5))
cal = HestonCalibration(pricer=pricer, vol_surface=vs, moneyness_weight=-0)
len(cal.options)
```

```{code-cell} ipython3
cal.model
```

```{code-cell} ipython3
cal.fit()
```

```{code-cell} ipython3
pricer.model
```

```{code-cell} ipython3
cal.plot(index=6, max_moneyness_ttm=1)
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
