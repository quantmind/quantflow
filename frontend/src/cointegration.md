---
title: Cointegration
---

# Cointegration Analysis of Cryptocurrencies

Johansen cointegration test on BTC, ETH, and SOL log prices. The residual (spread) of the first cointegrating vector is plotted below.

```js
import {fetchJson} from "./lib/api.js";
import * as Plot from "npm:@observablehq/plot";
```

```js
const frequencyInput = Inputs.select(["daily", "4hour", "1hour", "30min", "15min", "5min", "1min"], {label: "Frequency", value: "daily"});
const frequency = Generators.input(frequencyInput);
```

```js
display(frequencyInput);
```

```js
await new Promise(r => setTimeout(r, 300));
const data = await fetchJson(`/.api/cointegration?frequency=${frequency}`);
```

```js
display(html`<p>Cointegrating vector (BTC, ETH, SOL): <strong>[${data.deltas.map(d => d.toFixed(4)).join(", ")}]</strong></p>`);
```

```js
const residuals = data.dates.map((d, i) => ({date: new Date(d), residual: data.residuals[i]}));

display(Plot.plot({
  width: 900,
  height: 400,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Date", type: "utc", grid: true},
  y: {label: "Residual (Spread)", grid: true},
  marks: [
    Plot.line(residuals, {x: "date", y: "residual", stroke: "var(--theme-foreground-focus)", strokeWidth: 1.5, tip: true}),
    Plot.ruleY([0], {stroke: "var(--theme-foreground-muted)", strokeDasharray: "4,4"}),
  ]
}));
```

## Why Pick the Largest Eigenvalue?

In the Johansen cointegration test, the eigenvalues are sorted in descending order, and each one corresponds to a different potential cointegrating vector. The magnitude of the eigenvalue represents the strength and stability of the corresponding cointegrating relationship.

1. **Strongest Relationship:** The largest eigenvalue corresponds to the linear combination of the time series that is "most stationary." The resulting spread has the strongest tendency to revert to its mean.
2. **Statistical Significance:** The test statistics (Trace and Maximum Eigenvalue tests) are functions of these eigenvalues, helping determine how many significant cointegrating relationships exist.
3. **Practical Application:** For pairs trading, we want the most reliable long-run equilibrium. The vector associated with the largest eigenvalue gives the most mean-reverting portfolio.

## Normalization of the Cointegrating Vector

The Johansen test normalizes eigenvectors with respect to the cross-product matrix S11 of the input series, not the identity matrix. This means the Euclidean norm of the returned eigenvector depends on the scale of the input data and can be arbitrarily large.

To obtain a unit-norm vector that applies directly to raw log-prices, the API does the following:

1. **Standardize inputs:** each log-price series is divided by its standard deviation before running the Johansen test. This makes S11 approximately equal to the identity matrix, so the eigenvectors come out with unit Euclidean norm in the scaled space.
2. **Rescale back:** the eigenvector is divided by the same standard deviations to convert the weights back into log-price space.
3. **Re-normalize:** the rescaled vector is divided by its L2 norm to restore unit length.

The resulting vector `[δ_BTC, δ_ETH, δ_SOL]` can be applied directly to raw log-prices to compute the spread:

```tex
\text{spread}(t) = \delta_\text{BTC} \ln P_\text{BTC}(t) + \delta_\text{ETH} \ln P_\text{ETH}(t) + \delta_\text{SOL} \ln P_\text{SOL}(t)
```

The cointegrating direction is only defined up to a scalar multiple, so the final L2 normalization is statistically valid and ensures the vector components are comparable across different time periods or asset sets.

## Should You Use Log Prices?

Yes, using log prices is generally recommended for cointegration analysis in finance:

1. **Percentage vs. Absolute Changes:** Log prices work with relative percentage changes rather than absolute dollar amounts, which is crucial when assets trade at vastly different scales.
2. **Variance Stabilization:** Log transformation helps stabilize variance in heteroscedastic financial data.
3. **Linearization:** Real-world economic relationships between assets are often multiplicative; logarithms convert these into linear additive relationships suitable for the Johansen test.
