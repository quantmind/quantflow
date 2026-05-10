# Release Instructions

Releases are driven by `v*` git tags. Pushing a tag triggers
`.github/workflows/release.yml`, which runs lint and the test suite, publishes
the package to PyPI (`make publish`), then extracts the matching `## vX.Y.Z`
section from `docs/release-notes.md` and publishes it as the GitHub Release
body.

## Cutting a release

1. Bump `version` in `pyproject.toml` to the new release version.
2. Add a `## vX.Y.Z` section at the top of `docs/release-notes.md` with the
   notes for the release. The header text is matched verbatim by the
   workflow's `awk` extractor, so it must be `## vX.Y.Z` exactly (no trailing
   dash, no title after the version). The release workflow fails if this
   section is missing.
3. Commit and merge to `main`; let the `build` workflow pass.
4. From `main`, run `make release` — it reads the version from
   `pyproject.toml`, asks for confirmation, then creates an annotated `vX.Y.Z`
   tag and pushes it. The `release` workflow takes it from there.

Do not publish to PyPI manually, and do not revive the old
`head_commit.message == 'release'` flow — the tag-triggered workflow is the
only supported path.

## Release-notes conventions

The `## vX.Y.Z` section in `docs/release-notes.md` is rendered both on the
docs site and as the GitHub Release body, so it must follow these conventions:

- Open with a one-paragraph summary describing the theme of the release. If
  the release contains breaking changes, point readers to the **Breaking
  changes** section in that paragraph.
- Group entries under H3 subsections in this order: `### Breaking changes`,
  `### New features`, `### Improvements and fixes`,
  `### Documentation and assets`. Omit any subsection that has no entries.
- Every PR reference must be a markdown link of the form
  `[#NN](https://github.com/quantmind/quantflow/pull/NN)` — never a bare
  `(#NN)`. GitHub's auto-linking only works in some contexts, and the explicit
  URL works everywhere. When one entry references multiple PRs, list them
  comma-separated inside one set of parentheses, each as its own link.
- Build the PR list by running `git log vPREV..HEAD --oneline` against the
  previous release tag and following each squashed-merge commit back to its
  PR. Cross-check with `gh pr list --state merged --base main` for any PRs
  merged since the previous tag.
- End the section with a
  `[Full changelog](https://github.com/quantmind/quantflow/compare/vPREV...vX.Y.Z)`
  link comparing the new tag against the previous one.
