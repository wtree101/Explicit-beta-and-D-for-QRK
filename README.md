# Streaming QRK analysis and experiments

This repository contains the canonical numerical theory and Monte Carlo
experiments for the paper on quantile randomized Kaczmarz (QRK) for streaming
corrupted linear systems.

The maintained Python packages are:

- `qrk_analysis`: produces theoretical curves. Includes quantiles, noise models, feasibility checks, parameter
  searches, and the integer `smallest_D` API.
- `experiments.heatmaps`: produces practical results through streaming
  simulations, heatmap generation, theoretical model selection, and stable
  text output.
- `heatmap_data_display`: renders explicitly selected cached heatmap data
  without rerunning simulations.

Archived exploratory programs live in `legacy/` and are not part of the
supported API.

## Setup and verification

Run from this directory in the Conda base environment:

```bash
conda activate base
python -m pip install -e .
python -m pytest tests qrk_analysis/tests -q
```

Python 3.10 or newer is required.

## Numerical bounds

The public upper-bound API returns the smallest certified integer subsample
size together with the selected slack parameters and diagnostics:

```python
from functools import partial

from qrk_analysis.feasibility.check import (
    check_feasibility_conditions_C_sup_revised,
)
from qrk_analysis.upper_bound import smallest_D

fixed_supremum = partial(
    check_feasibility_conditions_C_sup_revised,
    num_grid_Q=10,
    C_min=0.0,
    C_max=20.0,
    num_points_C=200,
)

result = smallest_D(
    beta=0.01,
    T=20_000,
    q=0.75,
    delta_f=0.1,
    D_max=1_000,
    c_target=0.0,
    num_grid=60,
    feasibility_check=fixed_supremum,
)
print(result["smallest_D"])
```

The result also contains `alpha_0`, `alpha_prime`, `c`, `p_l_c`, `p_u`,
`failure_prob`, and `hit_ceiling`. The legacy `D_precision` argument remains
accepted but does not affect the integer bisection.

## Paper-bound figures

Regenerate formal paper figures or exploratory Gaussian figures with:

```bash
python -m qrk_analysis.programs.demo_paper_bounds --paper --recompute
python -m qrk_analysis.programs.demo_paper_bounds --extra --recompute
python -m qrk_analysis.programs.demo_paper_bounds --paper --extra --recompute
```

With no group flag, the command defaults to `--paper`. Formal PDFs are written
to the parent paper's `PR_quantile/figures/` directory. Curve caches are kept
under `figure/paper_bounds/cache/`; Gaussian exploratory results are kept under
`figure/paper_extra/`.

## Heatmap experiments

Import simulation and generation functions directly from the public package:

```python
from experiments.heatmaps import (
    generate_heat_map_matrix,
    streaming_subsampled_qRK_step,
)
```

The supported experiment drivers are:

```bash
python heatmap_generation_D_vs_T_demo.py
python heatmap_generation_D_vs_beta_demo.py
python convergence_curves_D_demo.py
```

The first two write success matrices and theoretical boundaries to
`heat_map_raw_data/`; the convergence driver writes plots and `.npz` data to
`figure/`. Quantile diagnostics are written to `q_e/`.

Heatmap generation uses two independent contraction parameters:
`c_success` defines the empirical success criterion
`relative_error <= (1 - c_success / n) ** T`, and `c_theory` is passed as the
contraction target when computing the theoretical boundary. Generated data
filenames record both values.

Plot existing heatmap data separately from generation:

```bash
python -m heatmap_data_display.plot_heatmaps --list-profiles
python -m heatmap_data_display.plot_heatmaps --profile d-vs-t-massart
python -m heatmap_data_display.plot_heatmaps --profile d-vs-t-oblivious
python -m heatmap_data_display.plot_heatmaps --profile d-vs-beta-massart
python -m heatmap_data_display.plot_heatmaps --profile d-vs-beta-oblivious
python -m heatmap_data_display.plot_heatmaps --profile d-vs-beta-massart --d-max 80
python -m heatmap_data_display.plot_heatmaps --profile d-vs-beta-massart --color-scale threshold --color-center 0.9
```

Previews are written to `figure/heatmaps/`. An explicit `--paper` is required
to publish stable PDFs to the paper tree. Profiles always name exact cached
files; the plotting command never searches for the latest dataset. The
optional `--d-min` and `--d-max` change only the displayed range and default
to the full range in the cached data. Color scaling defaults to `linear`;
`--color-scale threshold --color-center 0.9` expands the color range near the
90% success threshold, while `--color-scale power --color-gamma 2` provides a
smooth power-law alternative.

[`docs/Experiment_setting.md`](docs/Experiment_setting.md) records the intended paper
heatmap configuration. Individual demos may temporarily use smaller grids,
horizons, or sample counts for exploration. The numerical comparison made
during the package merge is retained in [`docs/merge_audit.md`](docs/merge_audit.md).

Full heatmap sweeps and oblivious-noise bounds can be expensive. Use reduced
parameters for smoke testing, and preserve cached results for formal settings.
