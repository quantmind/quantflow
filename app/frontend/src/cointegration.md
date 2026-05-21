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
  x: {label: "Date", type: "utc"},
  y: {label: "Residual (Spread)"},
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

## Should You Use Log Prices?

Yes, using log prices is generally recommended for cointegration analysis in finance:

1. **Percentage vs. Absolute Changes:** Log prices work with relative percentage changes rather than absolute dollar amounts, which is crucial when assets trade at vastly different scales.
2. **Variance Stabilization:** Log transformation helps stabilize variance in heteroscedastic financial data.
3. **Linearization:** Real-world economic relationships between assets are often multiplicative; logarithms convert these into linear additive relationships suitable for the Johansen test.
