---
title: SuperSmoother & EWMA
---

# SuperSmoother & EWMA

Compare the SuperSmoother and EWMA filters applied to BTCUSD daily close prices.

```js
import {fetchJson} from "./lib/api.js";
import * as Plot from "npm:@observablehq/plot";
```

```js
const periodInput = Inputs.range([2, 100], {step: 1, value: 10, label: "Period"});
const period = Generators.input(periodInput);
```

```js
display(periodInput);
```

```js
await new Promise(r => setTimeout(r, 300));
const result = await fetchJson(`/.api/supersmoother?period=${period}`);
```

```js
const long = result.data.flatMap(d => [
  {date: d.date, price: d.close, signal: "Close"},
  {date: d.date, price: d.supersmoother, signal: "SuperSmoother"},
  {date: d.date, price: d.ewma, signal: "EWMA"},
]);

display(Plot.plot({
  width: 900,
  height: 450,
  marginLeft: 70,
  marginBottom: 50,
  style: {background: "transparent"},
  x: {label: "Date", type: "utc"},
  y: {label: "Price (USD)"},
  color: {legend: true, label: "Signal", domain: ["Close", "SuperSmoother", "EWMA"], range: ["#4c78a8", "#f58518", "#e45756"]},
  marks: [
    Plot.line(long, {x: "date", y: "price", stroke: "signal", strokeWidth: 1.5, tip: true}),
  ]
}));
```
