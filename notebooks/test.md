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

# Test

```{code-cell} ipython3
from quantflow.sp.weiner import Weiner
p = Weiner(0.5)
m = p.marginal(1)
m.std()
```

```{code-cell} ipython3
import plotly.express as px
N = 64
M = 8
dx = 4/N
alpha = 0.5
r = m.call_option(N, M, dx, alpha=alpha)
fig = px.line(r, x="x", y="y", markers=True)
fig.show()
```

```{code-cell} ipython3

```
