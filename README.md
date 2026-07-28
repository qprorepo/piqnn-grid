# PIQNN-Grid

**Physics-Informed Quantum Neural Networks for Real-Time Power Grid Transient Stability**

This repository contains the LaTeX source for the PIQNN-Grid manuscript: a physics-informed learning framework in which a parameterized quantum circuit, rather than a classical network, is trained against a physics-informed loss built directly from Kirchhoff's Laws and the multi-machine swing equations, for real-time transient-stability assessment of power grids.

---

## Overview

Classical transient-stability assessment solves the nonlinear differential-algebraic equations governing generator swing dynamics and the AC power-flow network — too slowly for the 20–200 ms control loops that modern grid protection requires. This work introduces **PIQNN-Grid**, which combines:

- The **Grid-Topology-Aware (GTA) Ansatz** — an entangling structure that mirrors the power network's own admittance graph, so inter-bus coupling in the swing and power-flow residuals maps onto physically motivated two-qubit interactions.
- The **Quantum Spectral Differentiation (QSD) Theorem** — exact time derivatives of a circuit's expectation-value outputs from a finite set of equally spaced circuit evaluations, removing the need for finite-difference approximation of the physics residual.
- The **Quantum Energy Operator (QEO)** — reconstructs a Lyapunov-style transient energy function directly from the network's readout qubits, enabling a dissipativity-based stability regularizer.
- A **Quantum Real-Time Feasibility Theorem** — bounds the shot budget required to meet a control-loop deadline at a target confidence level.

The manuscript reports results on both a fully simulated benchmark and an independently built pipeline evaluated against real IEEE 9-, 14-, and 30-bus MATPOWER case data, and reports both sets of numbers — including where the quantum classifier trails classical baselines — rather than only the more favorable result.

## Repository Structure

```
piqnn-grid/
├── main_fixed_1.tex     # Main manuscript source
├── glossary.tex         # Acronym and glossary definitions (glossaries package)
├── reference.bib        # Bibliography (biblatex/biber)
└── README.md
```

## Building the Manuscript

The document requires a standard TeX Live installation with `biblatex`/`biber` and the `glossaries` package.

```bash
pdflatex -interaction=nonstopmode main_fixed_1.tex
biber main_fixed_1
pdflatex -interaction=nonstopmode main_fixed_1.tex
pdflatex -interaction=nonstopmode main_fixed_1.tex
```

Three `pdflatex` passes (with a `biber` run in between) are required to correctly resolve all cross-references, citations, and glossary entries.

### Dependencies

- `amsmath`, `amssymb`, `amsthm`, `mathtools`, `bm`, `bbm`, `tensor`
- `biblatex` (biber backend), `glossaries`
- `tikz` (with `quantikz` for circuit diagrams), `pgfplots`
- `tcolorbox`, `titlesec`, `caption`, `geometry`

## Citing This Work

If you use this manuscript or its results, please cite it as:

```bibtex
@unpublished{piqnngrid2026,
  title  = {PIQNN-Grid: Physics-Informed Quantum Neural Networks for Power Grid Stability},
  author = {[Author Name]},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## License

This project is licensed under the terms described in [LICENSE](LICENSE). See that file for details.

## Contact

For questions regarding this manuscript, please open an issue in this repository.# piqnn-grid
