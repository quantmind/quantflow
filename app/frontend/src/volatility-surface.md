---
title: Volatility Surface
---

# Volatility Surface

Live implied volatility surface from market options data. Crypto assets (BTC, ETH) use [Deribit](https://www.deribit.com/); equities (SPY, AAPL, NVDA) use [Yahoo Finance](https://finance.yahoo.com/).

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
display(assetInput);
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
  implied_vol: parseFloat(d.implied_vol),
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

display(html`<button onclick=${downloadInputs} style="cursor: pointer">Download Inputs (JSON)</button>`);
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

```js
const smileData = selectedMaturity === null
  ? enriched
  : enriched.filter(d => d.maturity === selectedMaturity);

const xLabel = {moneyness: "Moneyness (log(K/F) / √T)", log_strike: "Log Strike (log K/F)", strike: "Strike"}[xAxis];

display(Plot.plot({
  width: 800,
  height: 450,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: xLabel},
  y: {label: "Implied Volatility", percent: true},
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
      y: "implied_vol",
      fill: "maturity",
      r: 3,
      opacity: 0.8,
      tip: true
    }),
    Plot.ruleY([0]),
    ...(xAxis === "moneyness" ? [Plot.ruleX([0], {stroke: "var(--theme-foreground-muted)", strokeDasharray: "4,4"})] : [])
  ]
}));
```

## Volatility Term Structure

```js
// ATM vol per maturity (closest to moneyness = 0)
const atmByMaturity = maturities.map(m => {
  const slice = enriched.filter(d => d.maturity === m);
  const atm = slice.reduce((best, d) => Math.abs(d.moneyness) < Math.abs(best.moneyness) ? d : best);
  return {maturity: m, implied_vol: atm.implied_vol};
});

display(Plot.plot({
  width: 800,
  height: 350,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Maturity", type: "point"},
  y: {label: "ATM Implied Volatility", percent: true},
  marks: [
    Plot.line(atmByMaturity, {x: "maturity", y: "implied_vol", stroke: "var(--theme-foreground-focus)", strokeWidth: 2}),
    Plot.dot(atmByMaturity, {
      x: "maturity",
      y: "implied_vol",
      fill: "var(--theme-foreground-focus)",
      r: 5,
      tip: true
    })
  ]
}));
```

## Discount Curves

```js
const quoteCurve = data.quote_curve.ttm.map((t, i) => ({ttm: t, rate: data.quote_curve.rates[i], curve: "Quote"}));
const assetCurve = data.asset_curve.ttm.map((t, i) => ({ttm: t, rate: data.asset_curve.rates[i], curve: "Asset"}));
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
