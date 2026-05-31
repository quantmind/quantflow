---
name: quantflow-bibliography-instructions
description: 'Instructions for bibliography in quantflow'
applyTo: '/**'
---

Instructions for updating the bibliography for quantflow:

* The bibliography for quantflow is maintained in `docs/references.bib`
* Each entry must be added in alphabetical order of the label (the part after the first `{` in the bib entry).
* Regenerate the markdown file `docs/bibliography.md` from the `bib` file using the
  ```bash
  make docs-bib
  ```
* Run this command every time you update the `bib` file to keep the markdown file in sync
* Do not edit `docs/bibliography.md` directly as it is generated from `docs/references.bib`.
