---
title: Hurst Exponent
---

# Hurst Exponent

The [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent) is a statistical measure used to uncover the long-term memory of a time series. It helps determine if a financial asset is purely random (H = 0.5), trending (H > 0.5), or mean-reverting (H < 0.5).

```js
import {fetchJson} from "./lib/api.js";
import * as Plot from "npm:@observablehq/plot";
```

## Wiener Process

We sample a Wiener process path over one day (one time step per second) and estimate the Hurst exponent from both realized variance and range-based OHLC estimators.

```js
const sigmaInput = Inputs.range([0.1, 10], {step: 0.1, value: 2.0, label: "Sigma (volatility)"});
const sigma = Generators.input(sigmaInput);
```

```js
display(sigmaInput);
```

```js
await new Promise(r => setTimeout(r, 300));
const wiener = await fetchJson(`/.api/hurst-wiener?sigma=${sigma}`);
```

```js
display(html`<div style="display: flex; gap: 2rem; flex-wrap: wrap">
  <div><strong>Realized Std:</strong> ${wiener.realized_std.toFixed(4)}</div>
  <div><strong>Hurst (realized):</strong> ${wiener.hurst_exponent.toFixed(4)}</div>
  <div><strong>Hurst (Parkinson):</strong> ${wiener.ohlc_hurst_pk.toFixed(4)}</div>
  <div><strong>Hurst (Garman-Klass):</strong> ${wiener.ohlc_hurst_gk.toFixed(4)}</div>
  <div><strong>Hurst (Rogers-Satchell):</strong> ${wiener.ohlc_hurst_rs.toFixed(4)}</div>
</div>`);
```

### Range-based Volatility Estimators

Volatility estimated from OHLC data using Parkinson (1980), Garman & Klass (1980), and Rogers & Satchell (1991) estimators across different sampling periods.

```js
const estData = wiener.estimator_periods.flatMap((p, i) => [
  {period: p, value: wiener.estimator_pk[i], estimator: "Parkinson"},
  {period: p, value: wiener.estimator_gk[i], estimator: "Garman-Klass"},
  {period: p, value: wiener.estimator_rs[i], estimator: "Rogers-Satchell"},
]);

display(Plot.plot({
  width: 800,
  height: 400,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Sampling Period", type: "point"},
  y: {label: "Estimated Volatility"},
  color: {legend: true, label: "Estimator"},
  marks: [
    Plot.dot(estData, {x: "period", y: "value", fill: "estimator", r: 5, tip: true}),
    Plot.ruleY([sigma], {stroke: "var(--theme-foreground-muted)", strokeDasharray: "4,4"}),
  ]
}));
```

## Mean-Reverting (Vasicek)

For mean-reverting processes, the Hurst exponent is less than 0.5. Higher mean reversion (kappa) leads to a lower Hurst exponent.

```js
const kappaInput = Inputs.range([1, 500], {step: 1, value: 10, label: "Kappa (mean reversion)"});
const kappa = Generators.input(kappaInput);
```

```js
display(kappaInput);
```

```js
await new Promise(r => setTimeout(r, 300));
const vasicek = await fetchJson(`/.api/hurst-vasicek?kappa=${kappa}`);
```

```js
display(html`<div style="display: flex; gap: 2rem; flex-wrap: wrap">
  <div><strong>Hurst (realized):</strong> ${vasicek.hurst_realized.toFixed(4)}</div>
  <div><strong>Hurst (Parkinson):</strong> ${vasicek.hurst_pk.toFixed(4)}</div>
  <div><strong>Hurst (Garman-Klass):</strong> ${vasicek.hurst_gk.toFixed(4)}</div>
  <div><strong>Hurst (Rogers-Satchell):</strong> ${vasicek.hurst_rs.toFixed(4)}</div>
</div>`);
```

```js
const vasicekPath = vasicek.dates.map((d, i) => ({date: new Date(d), value: vasicek.values[i]}));

display(Plot.plot({
  width: 800,
  height: 350,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Time", type: "utc"},
  y: {label: "Value"},
  marks: [
    Plot.line(vasicekPath, {x: "date", y: "value", stroke: "var(--theme-foreground-focus)", strokeWidth: 1.5}),
  ]
}));
```
