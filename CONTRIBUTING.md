# Contributing to PIQNN-Grid

Thanks for your interest in this project. This repository holds the LaTeX
source for an active research manuscript, so contribution norms are a little
different from a typical software project.

## Ways to contribute

- **Typos, grammar, and clarity fixes** — open a pull request directly.
- **Errors in derivations, equations, or numerical results** — please open an
  issue first with the specific location (section/equation/table number) and
  a description of the problem, so it can be verified before any text changes.
- **Reproducibility issues** (build failures, missing dependencies) — open an
  issue with your OS, TeX distribution, and the full compiler log.
- **New citations or references** — open an issue naming the claim that needs
  a source; please do not open a PR that adds a `\cite{}` and a BibTeX entry
  without first confirming the source is accurate and directly supports the
  claim.

## Before opening a pull request

1. **Build locally first.** Run the full three-pass build (see the
   [README](README.md#building-the-manuscript) or `make pdf`) and confirm
   there are no new LaTeX errors, undefined references, or undefined
   citations introduced by your change.
2. **Keep diffs minimal and scoped.** One concern per pull request — a
   wording fix and a numerical correction should be separate PRs.
3. **Do not renumber or relabel** existing equations, theorems, figures, or
   tables unless the PR is specifically about restructuring, since other
   `\cref`/`\ref` targets and any external references depend on them.
4. **Glossary terms:** if you introduce a new acronym or technical term that
   is used more than once, add it to `glossary.tex` via `\newacronym{...}`
   and reference it with `\gls{...}`/`\glspl{...}` rather than typing the
   acronym in plain text.

## Style conventions used in this manuscript

- Citations: `\cite{key}` (biblatex, numeric style, `sorting=none`).
- Cross-references: `\cref{...}` / `\Cref{...}` (never bare `\ref{...}`).
- Glossary/acronyms: `\gls{key}` / `\glspl{key}` on every occurrence — the
  `glossaries` package handles first-use expansion automatically.
- Theorem-like environments follow the `\begin{theorem}{Title}{label}` custom
  signature defined in the preamble — please match this when adding new
  results.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it.

## Questions

If anything above is unclear, open an issue and ask — that's better than
guessing and submitting a large, hard-to-review PR.
