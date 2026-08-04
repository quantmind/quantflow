---
title: Volatility Surface
---

# Volatility Surface

Live implied volatility surface from market options data. Crypto assets (BTC, ETH) use the [Deribit volatility surface loader](https://quantflow.quantmind.com/api/data/deribit/#quantflow.data.deribit.Deribit.volatility_surface_loader); equities (SPY, AAPL, NVDA) use the [Yahoo Finance volatility surface loader](https://quantflow.quantmind.com/api/data/yahoo/#quantflow.data.yahoo.Yahoo.volatility_surface_loader).

The forwards are calibrated from put call parity and price the options directly. The quote and asset discount curves are then derived from the calibrated forwards: the asset curve is an interpolated curve through the implied discount factors, while the quote curve is kept at no discounting for crypto assets, whose inverse options settle without discounting, and fitted as an interpolated curve for equity assets.

```js
import {fetchJson} from "./lib/api.js";
import {observeScheme, palette1} from "./lib/palette.js";
import * as Plot from "npm:@observablehq/plot";
import * as d3 from "npm:d3";
```

```js
const assetInput = Inputs.select(["BTC", "ETH", "SPY", "AAPL", "NVDA"], {label: "Asset", value: "BTC"});
const asset = Generators.input(assetInput);
```

```js
display(html`<div style="display: flex; gap: 1rem; align-items: end; flex-wrap: wrap">${assetInput}</div>`);
```

```js
const data = await fetchJson(`/.api/volatility-surface?asset=${asset}`);
```

```js
// Options come pre-computed from the API with all fields
const options = data.options;

// Unique maturities sorted by date
const maturities = [...new Set(options.map(d => d.maturity))].sort();

// Parse numeric fields (API returns Decimals as strings)
const enriched = options.map(d => ({
  ...d,
  strike: parseFloat(d.strike),
  forward: parseFloat(d.forward),
  log_strike: parseFloat(d.log_strike),
  moneyness: parseFloat(d.moneyness),
  ttm: parseFloat(d.ttm),
  iv: parseFloat(d.iv),
  price_bp: parseFloat(d.price_bp),
  open_interest: parseFloat(d.open_interest),
  volume: parseFloat(d.volume),
}));

// Get spot from inputs
const spotInput = data.inputs.inputs.find(d => d.security_type === "spot");
const spotMid = spotInput ? (parseFloat(spotInput.bid) + parseFloat(spotInput.ask)) / 2 : null;
```

```js
const refDate = new Date(data.inputs.quote_curve.ref_date);
const formatDate = d3.utcFormat("%d %b %Y %H:%M:%S UTC");
display(html`<p style="color: var(--theme-foreground); font-size: 1.1rem">${formatDate(refDate)} · Spot: <strong>${spotMid ? d3.format(",.0f")(spotMid) : "N/A"} USD</strong> · ${enriched.length} options across ${maturities.length} maturities</p>`);
```

```js
const downloadInputs = () => {
  const blob = new Blob([JSON.stringify(data.inputs, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `volsurface_${asset}_${d3.utcFormat("%Y%m%d_%H%M%S")(refDate)}.json`;
  a.click();
  URL.revokeObjectURL(url);
};

display(html`<button onclick=${downloadInputs} style="cursor: pointer; background: var(--qf-primary); color: #fff; border: none; padding: 0.5em 1em; border-radius: 4px;">Download Inputs (JSON)</button>`);
```

```js
const maturityInput = Inputs.select(
  [null, ...maturities],
  {label: "Maturity", value: null, format: d => d === null ? "All" : d.slice(0, 10)}
);
const selectedMaturity = Generators.input(maturityInput);

const xAxisInput = Inputs.select(
  ["moneyness", "log_strike", "strike"],
  {label: "X-Axis", value: "moneyness", format: d => ({moneyness: "Moneyness", log_strike: "Log Strike", strike: "Strike"}[d])}
);
const xAxis = Generators.input(xAxisInput);
```

```js
const scheme = Generators.observe(observeScheme(palette1));
```

```js
display(html`<div style="display: flex; gap: 1rem; align-items: end; flex-wrap: wrap">${maturityInput}${xAxisInput}</div>`);
```

## Volatility Smile

The dots are the market implied volatilities of the bid and offer option quotes at each strike.

The solid lines are the fitted [eSSVI](https://quantflow.quantmind.com/api/options/ssvi/) model: each maturity has its own ATM total variance ${tex`\theta`}, curvature ${tex`\psi`} and correlation ${tex`\rho`}, calibrated slice by slice so the surface is free of static arbitrage by construction.

```js
// eSSVI model fitted to the surface (theta, psi and rho per maturity)
const ssvi = {
  ttm: data.ssvi.ttm.map(parseFloat),
  theta: data.ssvi.theta.map(parseFloat),
  psi: data.ssvi.psi.map(parseFloat),
  rho: data.ssvi.rho.map(parseFloat)
};

// Fitted node closest to tau (nodes coincide with maturities)
function ssviNode(tau) {
  let best = 0;
  for (let i = 1; i < ssvi.ttm.length; ++i) {
    if (Math.abs(ssvi.ttm[i] - tau) < Math.abs(ssvi.ttm[best] - tau)) best = i;
  }
  return best;
}

// eSSVI implied volatility at log-strike k = log(K/F) and maturity tau
function ssviIv(k, tau) {
  const node = ssviNode(tau);
  const theta = ssvi.theta[node];
  const rho = ssvi.rho[node];
  const pk = (ssvi.psi[node] / theta) * k;
  const w = 0.5 * theta * (1 + rho * pk + Math.sqrt((pk + rho) ** 2 + 1 - rho ** 2));
  return Math.sqrt(w / tau);
}

// Smooth SSVI smile per maturity across the observed log-strike range
const ssviData = maturities.flatMap(m => {
  const slice = enriched.filter(d => d.maturity === m);
  if (!slice.length) return [];
  const tau = slice[0].ttm;
  const forward = slice[0].forward;
  const ks = slice.map(d => d.log_strike);
  const kmin = Math.min(...ks), kmax = Math.max(...ks);
  const n = 80;
  return d3.range(n + 1).map(i => {
    const k = kmin + ((kmax - kmin) * i) / n;
    return {maturity: m, log_strike: k, moneyness: k / Math.sqrt(tau), strike: forward * Math.exp(k), iv: ssviIv(k, tau)};
  });
});
```

```js
const smileData = selectedMaturity === null
  ? enriched
  : enriched.filter(d => d.maturity === selectedMaturity);

const ssviSmile = selectedMaturity === null
  ? ssviData
  : ssviData.filter(d => d.maturity === selectedMaturity);

const xLabel = {moneyness: "Moneyness (log(K/F) / √T)", log_strike: "Log Strike (log K/F)", strike: "Strike"}[xAxis];

const ivValues = smileData.map(d => d.iv).filter(v => v > 0);
const ivMin = ivValues.length ? d3.min(ivValues) : 0;
const ivMax = ivValues.length ? d3.max(ivValues) : 1;
const ivPad = (ivMax - ivMin) * 0.1 || 0.05;

display(Plot.plot({
  width: 800,
  height: 450,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: xLabel},
  y: {label: "Implied Volatility", tickFormat: d3.format(".0%"), domain: [Math.max(0, ivMin - ivPad), ivMax + ivPad]},
  color: {
    type: "ordinal",
    domain: maturities,
    scheme: scheme,
    legend: selectedMaturity === null,
    label: "Maturity",
    tickFormat: d => d.slice(5, 10)
  },
  grid: true,
  marks: [
    Plot.dot(smileData, {
      x: xAxis,
      y: "iv",
      fill: "maturity",
      r: 3,
      opacity: 0.8,
      tip: true
    }),
    Plot.line(ssviSmile, {
      x: xAxis,
      y: "iv",
      z: "maturity",
      stroke: "maturity",
      strokeWidth: 1
    }),
    ...(xAxis === "moneyness" ? [Plot.ruleX([0], {stroke: "var(--theme-foreground-muted)", strokeDasharray: "4,4"})] : [])
  ]
}));
```

## eSSVI Parameters

The calibrated parameter term structure at each maturity node. The curvature is shown in standard deviation units, ${tex`\psi_\tau / 2\sqrt{\theta_\tau}`}, the ATM slope scale of the total standard deviation smile ${tex`s(k) = \sqrt{w(k)}`}: unlike the raw ${tex`\psi_\tau`}, which grows mechanically with the total variance, it is comparable across maturities. The correlation ${tex`\rho_\tau`} tilts the smile (negative values produce the put skew); the ATM slope of the std dev smile is their product.

```js
const ssviParams = ssvi.ttm.flatMap((t, i) => [
  {ttm: t, value: ssvi.psi[i] / (2 * Math.sqrt(ssvi.theta[i])), param: "Std dev curvature ψ/2√θ"},
  {ttm: t, value: ssvi.rho[i], param: "Correlation ρ"}
]);

display(Plot.plot({
  width: 800,
  height: 350,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Time to Maturity (years)", grid: true},
  y: {label: "Parameter value", grid: true},
  color: {
    legend: true,
    domain: ["Std dev curvature ψ/2√θ", "Correlation ρ"],
    range: ["var(--qf-accent)", "var(--qf-primary)"]
  },
  marks: [
    Plot.line(ssviParams, {x: "ttm", y: "value", stroke: "param", strokeWidth: 2}),
    Plot.dot(ssviParams, {x: "ttm", y: "value", fill: "param", r: 4, tip: true}),
    Plot.ruleY([0], {stroke: "var(--theme-foreground-muted)", strokeDasharray: "4,4"})
  ]
}));
```

## Volatility Term Structure

The dots are the market implied volatilities of the quotes closest to the money. The line is the fitted eSSVI ATM standard deviation ${tex`\sigma_{ATM}(\tau) = \sqrt{\theta_\tau / \tau}`}, which should track the market points closely.

```js
// ATM vol per maturity (closest to moneyness = 0) and the eSSVI ATM vol
const atmByMaturity = maturities.map(m => {
  const slice = enriched.filter(d => d.maturity === m);
  const atm = slice.reduce((best, d) => Math.abs(d.moneyness) < Math.abs(best.moneyness) ? d : best);
  const tau = slice[0].ttm;
  const node = ssviNode(tau);
  return {maturity: m, iv: atm.iv, model_iv: Math.sqrt(ssvi.theta[node] / tau)};
});

display(Plot.plot({
  width: 800,
  height: 350,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Maturity", type: "point"},
  y: {label: "ATM Implied Volatility", percent: true},
  color: {
    legend: true,
    domain: ["Market ATM", "eSSVI √(θ/τ)"],
    range: ["var(--qf-accent)", "var(--theme-foreground-focus)"]
  },
  marks: [
    Plot.line(atmByMaturity, {x: "maturity", y: "model_iv", stroke: "var(--theme-foreground-focus)", strokeWidth: 2}),
    Plot.dot(atmByMaturity, {
      x: "maturity",
      y: "iv",
      fill: "var(--qf-accent)",
      r: 5,
      tip: true
    })
  ]
}));
```

## Forward Curve

```js
const forwardCurve = data.forward_curve.ttm.map((t, i) => ({ttm: t, forward: data.forward_curve.forward[i]}));
const pcpForwards = data.pcp_forwards.map(d => ({ttm: d.ttm, forward: d.forward, maturity: d.maturity.slice(0, 10)}));

display(Plot.plot({
  width: 800,
  height: 350,
  marginLeft: 80,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Time to Maturity (years)", grid: true},
  y: {label: "Forward Price (USD)", grid: true},
  color: {legend: true, domain: ["Model", "Put-Call Parity"], range: ["var(--theme-foreground-focus)", "var(--qf-accent)"]},
  marks: [
    Plot.line(forwardCurve, {x: "ttm", y: "forward", stroke: "var(--theme-foreground-focus)", strokeWidth: 2}),
    Plot.dot(pcpForwards, {x: "ttm", y: "forward", fill: "var(--qf-accent)", r: 5, tip: true}),
  ]
}));
```

## Discount Curves

```js
const curveTypeLabels = {
  cir_curve: "CIR",
  nelson_siegel_curve: "Nelson-Siegel",
  vasicek_curve: "Vasicek",
  interpolated_linear_curve: "Interpolated (linear)",
  interpolated_monotonic_cubic_curve: "Interpolated (cubic)",
  no_discount_curve: "No discount",
};
const quoteLabel = `Quote - ${curveTypeLabels[data.quote_curve.curve.curve_type] ?? data.quote_curve.curve.curve_type}`;
const assetLabel = `Asset - ${curveTypeLabels[data.asset_curve.curve.curve_type] ?? data.asset_curve.curve.curve_type}`;
const quoteCurve = data.quote_curve.ttm.map((t, i) => ({ttm: t, rate: data.quote_curve.rates[i], curve: quoteLabel}));
const assetCurve = data.asset_curve.ttm.map((t, i) => ({ttm: t, rate: data.asset_curve.rates[i], curve: assetLabel}));
const curveData = [...quoteCurve, ...assetCurve];

display(Plot.plot({
  width: 800,
  height: 350,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Time to Maturity (years)"},
  y: {label: "Rate", percent: true},
  color: {legend: true, label: "Curve"},
  marks: [
    Plot.line(curveData, {x: "ttm", y: "rate", stroke: "curve", strokeWidth: 2}),
    Plot.ruleY([0], {stroke: "var(--theme-foreground-muted)", strokeDasharray: "4,4"})
  ]
}));
```
