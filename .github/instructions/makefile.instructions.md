---
applyTo: 'Makefile'
---

# Makefile Conventions

- Keep all targets sorted alphabetically.
- `help` target is the exception and should be the first target in the file.
- targets should be separated by a one blank line only.
- Each target should have a one-line description, starting with `##`, that describes what the target does. This description is used by the `help` target to generate documentation for all targets.
