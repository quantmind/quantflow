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

Sample the Gaussian OU (Vasicek) process for different mean reversion speeds ${tex`\kappa`} and number of paths. The process has unit volatility ${tex`\sigma = 1`} and initial value ${tex`x_0 = 0.5`}.

```js
const gKappaInput = Inputs.range([0.1, 5], {step: 0.1, value: 1.0, label: "kappa"});
const gKappa = Generators.input(gKappaInput);

const gSamplesInput = Inputs.range([100, 10000], {step: 100, value: 1000, label: "Samples"});
const gSamples = Generators.input(gSamplesInput);

const gAntitheticInput = Inputs.toggle({label: "Antithetic variates", value: true});
const gAntithetic = Generators.input(gAntitheticInput);
```

```js
display(html`<div style="display: flex; gap: 2rem; align-items: end; flex-wrap: wrap">${gKappaInput}${gSamplesInput}${gAntitheticInput}</div>`);
```

```js
await new Promise(r => setTimeout(r, 300));
const gaussianData = await fetchJson(`/.api/gaussian-sampling?kappa=${gKappa}&samples=${gSamples}&antithetic=${gAntithetic}`);
```

```js
const gBarData = gaussianData.x.map((x, i) => ({x, y: gaussianData.simulation[i], series: "Simulation"}));
const gLineData = gaussianData.x.map((x, i) => ({x, y: gaussianData.analytical[i], series: "Analytical"}));
const gBinWidth = gaussianData.x[1] - gaussianData.x[0];

display(Plot.plot({
  width: 800,
  height: 450,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Value"},
  y: {label: "Density"},
  color: {legend: true, domain: ["Simulation", "Analytical"], range: ["var(--theme-foreground-muted)", "var(--theme-foreground-focus)"]},
  marks: [
    Plot.rectY(gBarData, {x1: d => d.x - gBinWidth / 2, x2: d => d.x + gBinWidth / 2, y: "y", fill: "series", fillOpacity: 0.6, inset: 1, tip: true}),
    Plot.line(gLineData, {x: "x", y: "y", stroke: "series", strokeWidth: 2}),
  ]
}));

if (gaussianData.ks_statistic != null) {
  const ksColor = gaussianData.ks_pvalue < 0.05 ? "var(--theme-red)" : "var(--theme-green)";
  display(html`<p style="color: ${ksColor}; margin-top: 0.5rem;">
    KS statistic: <strong>${gaussianData.ks_statistic.toFixed(4)}</strong> &nbsp;|&nbsp;
    p-value: <strong>${gaussianData.ks_pvalue.toFixed(4)}</strong>
  </p>`);
}
```

## Poisson

Evaluate the Monte Carlo simulation for the Poisson process against the analytical PDF.

```js
const pIntensityInput = Inputs.range([2, 20], {step: 0.1, value: 2.0, label: "Intensity (λ)"});
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

display(Plot.plot({
  width: 800,
  height: 450,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Value"},
  y: {label: "Probability"},
  color: {legend: true, domain: ["Simulation", "Analytical"], range: ["var(--theme-foreground-muted)", "var(--theme-foreground-focus)"]},
  marks: [
    Plot.rectY(pBarSim, {x1: d => d.x - 0.3, x2: d => d.x - 0.05, y: "y", fill: "type", fillOpacity: 0.6, tip: true}),
    Plot.rectY(pBarAna, {x1: d => d.x + 0.05, x2: d => d.x + 0.3, y: "y", fill: "type", fillOpacity: 0.6, tip: true}),
  ]
}));

if (poissonData.chi2_statistic != null) {
  const chi2Color = poissonData.chi2_pvalue < 0.05 ? "var(--theme-red)" : "var(--theme-green)";
  display(html`<p style="color: ${chi2Color}; margin-top: 0.5rem;">
    χ² statistic: <strong>${poissonData.chi2_statistic.toFixed(4)}</strong> &nbsp;|&nbsp;
    p-value: <strong>${poissonData.chi2_pvalue.toFixed(4)}</strong>
  </p>`);
}
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
const deBarData = deData.x.map((x, i) => ({x, y: deData.simulation[i], series: "Simulation"}));
const deLineData = deData.x.map((x, i) => ({x, y: deData.analytical[i], series: "Analytical"}));
const deCharData = deData.char_x.map((x, i) => ({x, y: deData.char_y[i], series: "Char. function"}));
const deBinWidth = deData.x[1] - deData.x[0];

display(Plot.plot({
  width: 800,
  height: 450,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Value"},
  y: {label: "Density"},
  color: {legend: true, domain: ["Simulation", "Analytical", "Char. function"], range: ["var(--theme-foreground-muted)", "var(--theme-foreground-focus)", "var(--theme-accent)"]},
  marks: [
    Plot.rectY(deBarData, {x1: d => d.x - deBinWidth / 2, x2: d => d.x + deBinWidth / 2, y: "y", fill: "series", fillOpacity: 0.6, inset: 1, tip: true}),
    Plot.line(deLineData, {x: "x", y: "y", stroke: "series", strokeWidth: 2}),
    Plot.dot(deCharData, {x: "x", y: "y", fill: "series", r: 3}),
  ]
}));
```
