# Repository Guidelines

## Canonical ownership

- `qrk_analysis/` is the only maintained numerical theory implementation.
- `experiments/heatmaps/` is the only maintained heatmap API and owns
  simulation, generation, theory selection, and output formatting.
- `heatmap_data_display/` is the maintained cached-data plotting entry point.
  It reads explicit profiles and must not import or trigger simulations.
- Maintained code imports `qrk_analysis` and `experiments.heatmaps` directly;
  there are no top-level compatibility packages.
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
- Heatmap generation keeps `c_success` and `c_theory` independent.
  `c_success` defines
  `squared_relative_error <= (1 - c_success / n) ** T`; `c_theory` is the
  contraction target for the theoretical boundary. Preserve both values in
  generated filenames.
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

Plot existing heatmap data without recomputation with:

```bash
python -m heatmap_data_display.plot_heatmaps --list-profiles
python -m heatmap_data_display.plot_heatmaps --profile d-vs-t-massart
```

Profiles bind exact files under `heat_map_raw_data/`; never infer the latest
dataset. `--d-min` and `--d-max` affect only the displayed range and otherwise
default to the cached data extent. Preview PDFs go to `figure/heatmaps/`. Only
`--paper` may replace stable PDFs under the paper's `figures/heat_maps/`
directory. Validate every
selected input before writing, render each PDF for visual inspection, and
rebuild the paper after changing LaTeX figure references. Do not perform a
partial four-panel refresh, and keep shared display ranges aligned across the
Massart/oblivious comparison.
