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

# Fiscal Data

```{code-cell} ipython3
from quantflow.data.fiscal_data import FiscalData
```

```{code-cell} ipython3
fd = FiscalData()
```

```{code-cell} ipython3
data = await fd.securities()
data
```

```{code-cell} ipython3

```
