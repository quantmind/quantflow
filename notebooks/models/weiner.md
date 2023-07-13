---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Weiner Process

In this document, we use the term Weiner process $w_t$ to indicate a Brownian motion with standard deviation given by the parameter $\sigma$; that is to say, the one-dimensional Weiner process is defined as:

1. $w_t$ is Lévy process
2. $d w_t = w_{t+dt}-w_t \sim N\left(0, \sigma dt\right)$ where $N$ is the normal distribution

The characteristic exponent of $w$ is
\begin{equation}
    \phi_{w, u} = \frac{\sigma^2 u^2}{2}
\end{equation}

```{code-cell} ipython3
from quantflow.sp.weiner import WeinerProcess

pr = WeinerProcess(sigma=0.5)
pr
```

```{code-cell} ipython3
from quantflow.utils import plot
# create the marginal at time in the future
m = pr.marginal(1)
plot.plot_characteristic(m, n=32)
```

```{code-cell} ipython3
from quantflow.utils import plot
import numpy as np
plot.plot_marginal_pdf(m, 128)
```

## Test Option Pricing

```{code-cell} ipython3
from quantflow.options.pricer import OptionPricer
from quantflow.sp.weiner import WeinerProcess
pricer = OptionPricer(WeinerProcess(sigma=0.2))
pricer
```

```{code-cell} ipython3
import plotly.express as px
import plotly.graph_objects as go
from quantflow.options.bs import black_call
pricer.reset()
r = pricer.maturity(0.005)
b = r.black()
fig = px.line(x=r.moneyness_ttm, y=r.time_value, markers=True, title="Time value")
fig.add_trace(go.Scatter(x=b.moneyness_ttm, y=b.time_value, name=b.name, line=dict()))
fig.show()
```

```{code-cell} ipython3
pricer.maturity(0.1, alpha=8).plot()
```
