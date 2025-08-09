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

# Sampling Tools

The library use the `Paths` class for managing monte carlo paths.

```{code-cell}
from quantflow.utils.paths import Paths

nv = Paths.normal_draws(paths=1000, time_horizon=1, time_steps=1000)
```

```{code-cell}
nv.var().mean()
```

```{code-cell}
nv = Paths.normal_draws(paths=1000, time_horizon=1, time_steps=1000, antithetic_variates=False)
nv.var().mean()
```

```{code-cell}

```
