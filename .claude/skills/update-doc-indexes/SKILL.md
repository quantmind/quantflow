---
name: update-doc-indexes
description: Update top level API index pages with missing documented pages and classes
---

# Update API index pages

Update every top level API index page so it references all the relevant
documented pages and classes of its section.

## Scope

The top level index pages are `docs/api/<section>/index.md`, one per section
listed under `API Reference` in the `nav` of `mkdocs.yml`
(currently: data, dists, options, rates, sp, ta, utils).
If arguments name one or more sections, restrict the update to those.

## Procedure

For each section:

1. Read the section entry in the `nav` of `mkdocs.yml` and list its pages.
2. For each page, collect what it documents: the H1 title and the
   mkdocstrings entries (`::: fully.qualified.path` lines).
3. Read the section `index.md` and identify pages or classes that are not
   mentioned anywhere on it.
4. Add the missing entries to the index page:
   * Follow the existing layout of that index page. If it uses tables with
     `| Class | Description |` columns, extend or add tables in the same
     format. Only introduce a new H2 section when the missing entries do not
     fit an existing one, and match the style of the sections already there.
   * Ask the developer where to place a new section if the page layout does
     not make the placement obvious.
   * Link classes and functions with mkdocstrings cross-references
     (`[ClassName][fully.qualified.path]`) and pages with relative markdown
     links (e.g. `[SVI Volatility Smile](svi.md)`).
   * Write a one line description for each entry, based on the docstring or
     the page introduction. Link concepts to `docs/glossary.md` instead of
     redefining them.
5. Do not remove or rewrite existing content: this skill only adds what is
   missing and fixes references that are broken.

## Conventions

* Follow the documentation rules in `.github/copilot-instructions.md`
  (no dashes as punctuation, short paragraphs, relative links).
* Do not edit generated files (`docs/bibliography.md`, `readme.md`).
* After the update, list for the developer which entries were added to which
  index page.
