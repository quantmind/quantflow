---
title: Jump Diffusion Vol Surface
---

# Jump Diffusion Volatility Surface

Compare a Jump Diffusion volatility surface with a Heston Stochastic Volatility with Jumps model.

```js
import {fetchJson} from "./lib/api.js";
import * as Plot from "npm:@observablehq/plot";
```

```js
const modelInput = Inputs.select(new Map([["Jump Diffusion", "jd"], ["Heston with Jumps", "hj"]]), {label: "Model", value: "jd"});
const modelVal = Generators.input(modelInput);

const volInput = Inputs.range([0.1, 0.8], {step: 0.01, value: 0.4, label: "Long term volatility"});
const volVal = Generators.input(volInput);

const sigmaInput = Inputs.range([0.1, 2], {step: 0.1, value: 0.5, label: "Vol of vol"});
const sigmaVal = Generators.input(sigmaInput);

const kappaInput = Inputs.range([0.1, 2], {step: 0.1, value: 0.5, label: "Variance mean reversion"});
const kappaVal = Generators.input(kappaInput);

const rhoInput = Inputs.range([-0.6, 0.6], {step: 0.1, value: 0, label: "Correlation"});
const rhoVal = Generators.input(rhoInput);

const rInput = Inputs.range([0.6, 1.6], {step: 0.1, value: 1.0, label: "Initial vol ratio"});
const rVal = Generators.input(rInput);

const jfInput = Inputs.range([0.1, 0.9], {step: 0.05, value: 0.5, label: "Jump fraction"});
const jfVal = Generators.input(jfInput);

const jiInput = Inputs.range([10, 100], {step: 5, value: 10, label: "Jump intensity"});
const jiVal = Generators.input(jiInput);

const jaInput = Inputs.range([-2, 2], {step: 0.1, value: 0, label: "Jump asymmetry"});
const jaVal = Generators.input(jaInput);
```

```js
display(html`<div style="display: flex; flex-wrap: wrap; gap: 1rem; align-items: end;">
  ${modelInput}${volInput}${rInput}${sigmaInput}${kappaInput}${rhoInput}${jfInput}${jiInput}${jaInput}
</div>`);
```

```js
await new Promise(r => setTimeout(r, 300));
const params = new URLSearchParams({
  model: modelVal, vol: volVal, sigma: sigmaVal, kappa: kappaVal,
  rho: rhoVal, r: rVal, jump_fraction: jfVal, jump_intensity: jiVal, jump_asymmetry: jaVal
});
const data = await fetchJson(`/.api/heston-vol-surface?${params}`);
```

## Volatility Smile

```js
const flat = data.ttm.flatMap((t, i) =>
  data.moneyness.map((m, j) => ({ttm: t.toFixed(2), moneyness: m, implied_vol: data.implied_vol[i][j]}))
).filter(d => d.implied_vol > 0);

display(Plot.plot({
  width: 800,
  height: 450,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Moneyness"},
  y: {label: "Implied Volatility", percent: true},
  color: {type: "ordinal", scheme: "turbo", legend: true, label: "TTM"},
  marks: [
    Plot.line(flat, {x: "moneyness", y: "implied_vol", stroke: "ttm", strokeWidth: 1.5, tip: true}),
    Plot.ruleX([0], {stroke: "var(--theme-foreground-muted)", strokeDasharray: "4,4"}),
  ]
}));
```

## Volatility Term Structure

```js
const atmByTtm = data.ttm.map((t, i) => {
  const midIdx = Math.floor(data.moneyness.length / 2);
  return {ttm: t, implied_vol: data.implied_vol[i][midIdx]};
}).filter(d => d.implied_vol > 0);

display(Plot.plot({
  width: 800,
  height: 350,
  marginLeft: 60,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Time to Maturity"},
  y: {label: "ATM Implied Volatility", percent: true},
  marks: [
    Plot.line(atmByTtm, {x: "ttm", y: "implied_vol", stroke: "var(--theme-foreground-focus)", strokeWidth: 2}),
    Plot.dot(atmByTtm, {x: "ttm", y: "implied_vol", fill: "var(--theme-foreground-focus)", r: 4, tip: true}),
  ]
}));
```
