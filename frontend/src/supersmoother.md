---
title: SuperSmoother & EWMA
---

# SuperSmoother & EWMA

Both the [SuperSmoother](https://quantflow.quantmind.com/api/ta/supersmoother/) and [EWMA](https://quantflow.quantmind.com/api/ta/ewma/) are online filters that smooth noisy time series one observation at a time.
They share a single tuning knob (the period ${tex`p`}) so they can be compared directly.

The SuperSmoother is a two-pole Butterworth filter that removes high-frequency noise with very little lag.
EWMA applies exponential weighting with a decay derived from the same period, giving a simpler but slightly laggier result.

Use the slider below to see how the period affects both filters on BTCUSD daily closes.

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
const alpha = 1 - Math.exp(-1 / period);
const halfLife = period * Math.LN2;
display(html`<p><strong>EWMA α</strong> = ${alpha.toFixed(4)} · <strong>half-life</strong> = ${halfLife.toFixed(2)}</p>`);
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
  x: {label: "Date", type: "utc", grid: true},
  y: {label: "Price (USD)", grid: true},
  color: {legend: true, label: "Signal", domain: ["Close", "SuperSmoother", "EWMA"], range: ["#94a3b8", "#2563eb", "#dc2626"]},
  marks: [
    Plot.line(long, {x: "date", y: "price", stroke: "signal", strokeWidth: 1.5, tip: true}),
  ]
}));
```

## EWMA smoothing factor

The EWMA smoothing factor ${tex`\alpha`} is derived from the period:

```tex
\alpha = 1 - \exp\left(-\frac{1}{p}\right)
```

## Period and half-life

The half-life ${tex`h`} is the number of steps after which an observation's weight decays to half.
For an exponential decay with half-life ${tex`h`}, the sum of all weights equals ${tex`h / \ln 2`}.
This makes the period the continuous-time equivalent of the number of observations in a simple moving average:

```tex
p = \frac{h}{\ln 2}
```

The period is always larger than the half-life (by a factor of ${tex`1/\ln 2 \approx 1.44`}), which means
a period of 10 corresponds to a half-life of about 6.9 steps.
