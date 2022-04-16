# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import plotly.express as px
from quantflow.utils.marginal import Marginal1D


def chracteristic_fig(m: Marginal1D, N: int, max_frequency: float):
    f = m.frequency_space(N, max_frequency)
    c = m.characteristic(f)
    df = pd.concat(
        (
            pd.DataFrame(dict(frequency=f, characteristic=c.real, name="real")),
            pd.DataFrame(dict(frequency=f, characteristic=c.imag, name="iamg")),
        )
    )
    return px.line(df, x="frequency", y="characteristic", color="name", markers=True)


# %%
