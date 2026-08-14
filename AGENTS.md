# Repository Guidelines

## Canonical ownership

- `qrk_analysis/` is the only maintained numerical theory implementation.
- `qrk_adv/` and `heatmap_data_generation/` are compatibility facades. Never
  add formulas, searches, or simulation logic to them.
- `experiments/heatmaps/` owns simulation, generation, and output formatting.
- `legacy/` is read-only historical material.

## Environment and tests

Use the Conda `base` environment and run from this repository root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Keep numerical grids explicit. The formal paper configuration uses `q=0.75`,
`T=20000`, `delta_f=0.1`, `C in [0,20]`, 200 C points, and 10 conditional
quantile points. Gaussian exploration uses `sigma in [0.01,10]` with 20 points.

## Invariants

- Finite-horizon searches use strict interior slack parameters and binary
  search for the largest failure-feasible `alpha_prime`.
- `smallest_D` returns an integer and a flat diagnostic result.
- Revised oblivious checks use the scaled quantile interval required by the
  paper and the same scaled lower endpoint in the failure penalty.
- Heatmap success remains
  `squared_relative_error <= (1 - c / n) ** T` using the preset `c`.
- Preserve heatmap corruption types, filenames, matrix orientation, worker
  cap, and independent finite intervals for `oblivious_large`.

## Generated output

Do not regenerate full Monte Carlo heatmaps unless explicitly requested.
Paper-bound curves may be regenerated with:

```bash
python -m qrk_analysis.programs.demo_paper_bounds --paper --recompute
```

Formal PDFs are written to the parent paper tree; curve caches stay under
`figure/paper_bounds/cache/`.
