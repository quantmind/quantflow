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

+++ {"tags": []}

# Heston Model and Option Pricing

A very important example of time-changed Lévy process useful for option pricing is the Heston model. In this model the Lévy process is a standard Brownian motion, while the activity rate follows a CIR process. The leverage effect can be accomodated by correlating the two Brownian motions as the following equations illustrates:

\begin{aligned}
    d x_t &= d w_t \\
    d \nu_t &= \kappa\left(\theta - \nu_t\right) dt + \sigma\sqrt{\nu_t} d z_t \\
    {\mathbb E}\left[d w_t d z_t\right] &= \rho dt
\end{aligned}

This means that the characteristic function of $y_t=x_{\tau_t}$ can be represented as

\begin{aligned}
    \Phi_{y_t, u} & = {\mathbb E}\left[e^{i u y_t}\right] = {\mathbb L}_{\tau_t}^u\left(\frac{u^2}{2}\right) \\
     &= e^{-a_{t,u} - b_{t,u} \nu_0}
\end{aligned}

```{code-cell} ipython3
from quantflow.sp.heston import Heston
```

```{code-cell} ipython3
pri = Heston.create(vol=0.5, kappa=1, sigma=0.2, rho=-0.5)
```

```{code-cell} ipython3
pri.std_norm(1)
```

```{code-cell} ipython3
m = pri.marginal(1)
```

```{code-cell} ipython3
m.std()
```

```{code-cell} ipython3
m.characteristic(1)
```

```{code-cell} ipython3
m = p.marginal(1)
m.characteristic(0)
```

```{code-cell} ipython3
m.std()
```

```{code-cell} ipython3
import numpy as np
```

```{code-cell} ipython3
import plotly.express as px
N = 128*2*2
du = 0.1
delta = m.std()/50
freq = np.fft.rfftfreq(N)
freq
```

```{code-cell} ipython3
psi = m.characteristic(freq)
fig = px.line(psi)
fig.show()
```

```{code-cell} ipython3
len(freq)
```

```{code-cell} ipython3
d = np.ones(N//2+1)
d[0] = 0.5
d
```

```{code-cell} ipython3
-np.linspace(-0.5*N*delta, 0.5*N*delta, N, False)
```

```{code-cell} ipython3
psi = d*m.characteristic(freq)
psi
```

```{code-cell} ipython3
d = np.fft.irfft(psi)
d
```

```{code-cell} ipython3
import plotly.express as px
fig = px.line(d)
fig.show()
```

```{code-cell} ipython3

```
