---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Calibration

Early pointers

* https://github.com/rlabbe/filterpy
* [filterpy book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

+++

## Calibrating ABC

For calibration we use {cite:p}`ukf`.
Lets consider the Heston model as a test case

```{code-cell} ipython3
from quantflow.sp.heston import Heston

pr = Heston.create(vol=0.6, kappa=1.3, sigma=0.8, rho=-0.6)
pr.variance_process.is_positive
```

The Heston model is a classical example where the calibration of parameters requires to deal with the estimation of an unobserved random variable, the stochastic variance. The model can be discretized as follow:

\begin{align}
 d \nu_t &= \kappa\left(\theta -\nu_t\right) dt + \sigma \sqrt{\nu_t} d z_t \\
 d s_t &= -\frac{\nu_t}{2}dt + \sqrt{\nu_t} d w_t \\
 {\mathbb E}\left[d w_t d z_t\right] &= \rho dt
\end{align}

noting that

\begin{equation}
d z_t = \rho d w_t + \sqrt{1-\rho^2} d b_t
\end{equation}

which leads to

\begin{align}
d \nu_t &= \kappa\left(\theta -\nu_t\right) dt + \sigma \sqrt{\nu_t} \rho d w_t + \sigma \sqrt{\nu_t} \sqrt{1-\rho^2} d b_t \\
d s_t &= -\frac{\nu_t}{2}dt + \sqrt{\nu_t} d w_t \\
\end{align}

and finally

\begin{align}
d \nu_t &= \kappa\left(\theta -\nu_t\right) dt + \sigma \rho \frac{\nu_t}{2} dt + \sigma \sqrt{\nu_t} \sqrt{1-\rho^2} d b_t + \sigma \rho d s_t\\
d s_t &= -\frac{\nu_t}{2}dt + \sqrt{\nu_t} d w_t \\
\end{align}

Our problem is to find the *best* estimate of $\nu_t$ given by ths equation based on the observations $s_t$.

The Heston model is a dynamic model which can be represented by a state-space form: $X_t$ is the state while $Z_t$ is the observable

\begin{align}
X_{t+1} &= f\left(X_t, \Theta\right) + B^x_t\\
Z_t &= h\left(X_t, \Theta\right) + B^z_t \\
B^x_t &= {\cal N}\left(0, Q_t\right) \\
B^z_t &= {\cal N}\left(0, R_t\right) \\
\end{align}

$f$ is the *state transition equation* while $h$ is the *measurement equation*.

+++

the state equation is given by

\begin{align}
X_{t+1} &= \left[\begin{matrix}\kappa\left(\theta\right) dt \\ 0\end{matrix}\right] + 
\end{align}

```{code-cell} ipython3
[p for p in pr.variance_process.parameters]
```

```{code-cell} ipython3

```

## Calibration against historical timeseries

We calibrate the Heston model agais historical time series, in this case the measurement is the log change for a given frequency.

\begin{align}
F_t &= \left[\begin{matrix}1 - \kappa\theta dt \\ 0\end{matrix}\right] \\
Q_t &= \left[\begin{matrix}1 - \kappa\theta dt \\ 0\end{matrix}\right]  \\
z_t &= d s_t
\end{align}

The observation vector is given by
\begin{align}
x_t &= \left[\begin{matrix}\nu_t && w_t && z_t\end{matrix}\right]^T \\
\bar{x}_t = {\mathbb E}\left[x_t\right] &= \left[\begin{matrix}\nu_t && 0 && 0\end{matrix}\right]^T
\end{align}

```{code-cell} ipython3
from quantflow.data.fmp import FMP
frequency = "1min"
async with FMP() as cli:
    df = await cli.prices("ETHUSD", frequency)
df = df.sort_values("date").reset_index(drop=True)
df
```

```{code-cell} ipython3
import plotly.express as px
fig = px.line(df, x="date", y="close", markers=True)
fig.show()
```

```{code-cell} ipython3
import numpy as np
from quantflow.utils.volatility import parkinson_estimator, GarchEstimator
df["returns"] = np.log(df["close"]) - np.log(df["open"])
df["pk"] = parkinson_estimator(df["high"], df["low"])
ds = df.dropna()
dt = cli.historical_frequencies_annulaized()[frequency]
fig = px.line(ds["returns"], markers=True)
fig.show()
```

```{code-cell} ipython3
import plotly.express as px
from quantflow.utils.bins import pdf
df = pdf(ds["returns"], num=20)
fig = px.bar(df, x="x", y="f")
fig.show()
```

```{code-cell} ipython3
g1 = GarchEstimator.returns(ds["returns"], dt)
g2 = GarchEstimator.pk(ds["returns"], ds["pk"], dt)
```

```{code-cell} ipython3
import pandas as pd
yf = pd.DataFrame(dict(returns=g2.y2, pk=g2.p))
fig = px.line(yf, markers=True)
fig.show()
```

```{code-cell} ipython3
r1 = g1.fit()
r1
```

```{code-cell} ipython3
r2 = g2.fit()
r2
```

```{code-cell} ipython3
sig2 = pd.DataFrame(dict(returns=np.sqrt(g2.filter(r1["params"])), pk=np.sqrt(g2.filter(r2["params"]))))
fig = px.line(sig2, markers=False, title="Stochastic volatility")
fig.show()
```

```{code-cell} ipython3
class HestonCalibration:
    
    def __init__(self, dt: float, initial_std = 0.5):
        self.dt = dt
        self.kappa = 1
        self.theta = initial_std*initial_std
        self.sigma = 0.2
        self.x0 = np.array((self.theta, 0))
    
    def prediction(self, x):
        return np.array((x[0] + self.kappa*(self.theta - x[0])*self.dt, -0.5*x[0]*self.dt))
    
    def state_jacobian(self):
        """THe Jacobian of the state equation"""
        return np.array(((1-self.kappa*self.dt, 0),(-0.5*self.dt, 0)))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
c = HestonCalibration(dt)
c.x0
```

```{code-cell} ipython3
c.prediction(c.x0)
```

```{code-cell} ipython3
c.state_jacobian()
```
