---
title: Sampling
---

# Sampling

Monte Carlo sampling of stochastic processes compared against their analytical distributions.

```js
import {fetchJson} from "./lib/api.js";
import * as Plot from "npm:@observablehq/plot";
```

## Gaussian (Vasicek)

Sample the Gaussian OU (Vasicek) process for different mean reversion speeds and number of paths.

```js
const gKappaInput = Inputs.range([0.1, 5], {step: 0.1, value: 1.0, label: "Kappa (mean reversion)"});
const gKappa = Generators.input(gKappaInput);

const gSamplesInput = Inputs.range([100, 10000], {step: 100, value: 1000, label: "Samples"});
const gSamples = Generators.input(gSamplesInput);
```

```js
display(html`<div style="display: flex; gap: 2rem; align-items: end; flex-wrap: wrap">${gKappaInput}${gSamplesInput}</div>`);
```

```js
await new Promise(r => setTimeout(r, 300));
const gaussianData = await fetchJson(`/.api/gaussian-sampling?kappa=${gKappa}&samples=${gSamples}`);
```

```js
const gBarData = gaussianData.x.map((x, i) => ({x, y: gaussianData.simulation[i]}));
const gLineData = gaussianData.x.map((x, i) => ({x, y: gaussianData.analytical[i]}));
const gBinWidth = gaussianData.x[1] - gaussianData.x[0];

display(Plot.plot({
  width: 800,
  height: 450,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Value"},
  y: {label: "Density"},
  marks: [
    Plot.rectY(gBarData, {x1: d => d.x - gBinWidth / 2, x2: d => d.x + gBinWidth / 2, y: "y", fill: "var(--theme-foreground-muted)", fillOpacity: 0.6, tip: true}),
    Plot.line(gLineData, {x: "x", y: "y", stroke: "var(--theme-foreground-focus)", strokeWidth: 2}),
  ]
}));
```

## Poisson

Evaluate the Monte Carlo simulation for the Poisson process against the analytical PDF.

```js
const pIntensityInput = Inputs.range([2, 5], {step: 0.1, value: 2.0, label: "Intensity (λ)"});
const pIntensity = Generators.input(pIntensityInput);

const pSamplesInput = Inputs.range([100, 10000], {step: 100, value: 1000, label: "Samples"});
const pSamples = Generators.input(pSamplesInput);
```

```js
display(html`<div style="display: flex; gap: 2rem; align-items: end; flex-wrap: wrap">${pIntensityInput}${pSamplesInput}</div>`);
```

```js
await new Promise(r => setTimeout(r, 300));
const poissonData = await fetchJson(`/.api/poisson-sampling?intensity=${pIntensity}&samples=${pSamples}`);
```

```js
const pBarSim = poissonData.x.map((x, i) => ({x, y: poissonData.simulation[i], type: "Simulation"}));
const pBarAna = poissonData.x.map((x, i) => ({x, y: poissonData.analytical[i], type: "Analytical"}));
const pBinWidth = poissonData.x.length > 1 ? poissonData.x[1] - poissonData.x[0] : 1;

display(Plot.plot({
  width: 800,
  height: 450,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Value"},
  y: {label: "Probability"},
  color: {legend: true, label: "Source"},
  marks: [
    Plot.barY(pBarSim, {x: "x", y: "y", fill: "var(--theme-foreground-muted)", fillOpacity: 0.6, dx: -2, tip: true}),
    Plot.barY(pBarAna, {x: "x", y: "y", fill: "var(--theme-foreground-focus)", fillOpacity: 0.6, dx: 2}),
  ]
}));
```

## Double Exponential

Sample the Asymmetric Laplace (double exponential) distribution with mean 0 and variance 1, parameterised by the asymmetry parameter κ.

```js
const deLogKappaInput = Inputs.range([-2, 2], {step: 0.1, value: 0.1, label: "log(κ) asymmetry"});
const deLogKappa = Generators.input(deLogKappaInput);

const deSamplesInput = Inputs.range([100, 10000], {step: 100, value: 1000, label: "Samples"});
const deSamples = Generators.input(deSamplesInput);
```

```js
display(html`<div style="display: flex; gap: 2rem; align-items: end; flex-wrap: wrap">${deLogKappaInput}${deSamplesInput}</div>`);
```

```js
await new Promise(r => setTimeout(r, 300));
const deData = await fetchJson(`/.api/double-exponential-sampling?log_kappa=${deLogKappa}&samples=${deSamples}`);
```

```js
const deBarData = deData.x.map((x, i) => ({x, y: deData.simulation[i]}));
const deLineData = deData.x.map((x, i) => ({x, y: deData.analytical[i]}));
const deCharData = deData.char_x.map((x, i) => ({x, y: deData.char_y[i]}));
const deBinWidth = deData.x[1] - deData.x[0];

display(Plot.plot({
  width: 800,
  height: 450,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Value"},
  y: {label: "Density"},
  marks: [
    Plot.rectY(deBarData, {x1: d => d.x - deBinWidth / 2, x2: d => d.x + deBinWidth / 2, y: "y", fill: "var(--theme-foreground-muted)", fillOpacity: 0.6, tip: true}),
    Plot.line(deLineData, {x: "x", y: "y", stroke: "var(--theme-foreground-focus)", strokeWidth: 2}),
    Plot.dot(deCharData, {x: "x", y: "y", fill: "var(--theme-accent)", r: 3}),
  ]
}));
```
