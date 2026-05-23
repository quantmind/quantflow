---
title: Yield Curve
---

# Yield Curve

Fit a yield curve to a set of interest rates. Drag the points to change the input rates.

```js
import {fetchJson} from "./lib/api.js";
import * as Plot from "npm:@observablehq/plot";
import * as d3 from "npm:d3";
```

```js
const curveType = view(Inputs.select(["nelson_siegel", "vasicek_curve", "cir_curve"], {label: "Curve type"}));
```

```js
const defaultRates = [
  {ttm: 1/365, rate: 0.043},
  {ttm: 7/365, rate: 0.043},
  {ttm: 1/12, rate: 0.044},
  {ttm: 0.25, rate: 0.045},
  {ttm: 0.5, rate: 0.046},
  {ttm: 1, rate: 0.047},
  {ttm: 2, rate: 0.043},
  {ttm: 3, rate: 0.041},
  {ttm: 5, rate: 0.040},
  {ttm: 7, rate: 0.041},
  {ttm: 10, rate: 0.043},
  {ttm: 20, rate: 0.048},
  {ttm: 30, rate: 0.049}
];

const inputRates = Mutable(defaultRates);
const setInputRates = (v) => inputRates.value = v;
```

```js
const params = new URLSearchParams();
for (const {ttm, rate} of inputRates) {
  params.append("ttm", ttm);
  params.append("rates", rate);
}
params.set("curve_type", curveType);
params.set("max_ttm", "30");
params.set("num_points", "200");

const result = await fetchJson(`/.api/yield-curve?${params}`);
```

```js
const fittedData = result.ttm
  .map((t, i) => ({ttm: t, rate: result.rates[i]}))
  .filter(d => d.ttm >= 1/365);
```

```js
const width = 640;
const height = 400;
const marginTop = 30;
const marginRight = 20;
const marginBottom = 40;
const marginLeft = 50;

const x = d3.scaleLog()
  .domain([1/365, 32])
  .range([marginLeft, width - marginRight]);

const y = d3.scaleLinear()
  .domain([0.02, 0.06])
  .range([height - marginBottom, marginTop]);

const svg = d3.create("svg")
  .attr("width", width)
  .attr("height", height)
  .attr("viewBox", [0, 0, width, height])
  .attr("style", "max-width: 100%; height: auto;");

svg.append("g")
  .attr("transform", `translate(0,${height - marginBottom})`)
  .call(d3.axisBottom(x)
    .tickValues([1/365, 1/52, 1/12, 0.25, 0.5, 1, 2, 5, 10, 30])
    .tickFormat(d => d < 1/12 ? `${Math.round(d*365)}d` : d < 1 ? `${Math.round(d*12)}m` : `${d}y`)
  )
  .append("text")
  .attr("x", width / 2)
  .attr("y", 35)
  .attr("fill", "currentColor")
  .attr("text-anchor", "middle")
  .text("Time to Maturity (years)");

svg.append("g")
  .attr("transform", `translate(${marginLeft},0)`)
  .call(d3.axisLeft(y).tickFormat(d3.format(".1%")))
  .append("text")
  .attr("x", -marginLeft + 10)
  .attr("y", marginTop - 15)
  .attr("fill", "currentColor")
  .attr("text-anchor", "start")
  .text("Rate");

const line = d3.line()
  .x(d => x(d.ttm))
  .y(d => y(d.rate));

const curvePath = svg.append("path")
  .datum(fittedData)
  .attr("fill", "none")
  .attr("stroke", "var(--qf-primary)")
  .attr("stroke-width", 2)
  .attr("d", line);

const points = [...inputRates];

svg.selectAll("circle")
  .data(points)
  .join("circle")
  .attr("cx", d => x(d.ttm))
  .attr("cy", d => y(d.rate))
  .attr("r", 7)
  .attr("fill", "var(--qf-accent)")
  .attr("cursor", "ns-resize")
  .call(d3.drag()
    .on("drag", function(event, d) {
      const newRate = y.invert(event.y);
      const clamped = Math.max(0.001, Math.min(0.1, newRate));
      d.rate = clamped;
      d3.select(this).attr("cy", y(clamped));
    })
    .on("end", function() {
      setInputRates(points.map(p => ({ttm: p.ttm, rate: p.rate})));
    })
  );

display(svg.node());
```

```js
display(result.curve);
```