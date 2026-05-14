# Yahoo

Fetch equity option chains from [Yahoo Finance](https://finance.yahoo.com/).

The client is intentionally minimal: it fetches a full option chain via the
public `v7/finance/options` endpoint and exposes a helper to build a
[VolSurfaceLoader][quantflow.options.surface.VolSurfaceLoader] from it.

You can import the module via

```python
from quantflow.data.yahoo import Yahoo
```

## Authentication

Yahoo Finance requires a session cookie and a `crumb` token for the options
endpoint. The client fetches both on the first request and caches the crumb
for the lifetime of the instance.

::: quantflow.data.yahoo.Yahoo
