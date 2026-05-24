---
name: quantflow-observable-instructions
description: 'Instructions for Observable Framework pages in quantflow'
applyTo: '/frontend/src/**'
---


# Observable Framework Instructions

These instructions apply to markdown files in `frontend/src/`, which are rendered by Observable Framework.

## Math

Observable Framework uses KaTeX for math rendering. The syntax is different from standard markdown math (no `$...$` or `$$...$$`).

- **Display math** (block, centered): use a `tex` fenced code block:

  ````
  ```tex
  E = mc^2
  ```
  ````

- **Inline math**: use the `tex` tagged template literal inside an inline expression:

  ```
  The smoothing factor ${tex`\alpha`} controls decay.
  ```

- Do not use `$...$` or `$$...$$` for math in Observable Framework pages.
- Do not use `\begin{equation}...\end{equation}` (that convention applies only to mkdocs pages in `docs/`).

## Downloads

Always render download actions as a `<button>` element (not an `<a>` link). Use the pattern:

```js
const downloadData = () => {
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "filename.json";
  a.click();
  URL.revokeObjectURL(url);
};

display(html`<button onclick=${downloadData} style="cursor: pointer; background: var(--qf-primary); color: #fff; border: none; padding: 0.5em 1em; border-radius: 4px;">Download (JSON)</button>`);
```
