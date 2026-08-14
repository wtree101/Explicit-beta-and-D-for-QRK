# Streaming QRK analysis and experiments

This repository is the canonical implementation for the explicit corruption
and subsample-size bounds in the streaming quantile randomized Kaczmarz paper.
It also contains the Monte Carlo heatmap experiments used to compare theory
with empirical convergence.

## Setup

Use Python 3.10 or newer. In the project Conda environment:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Packages

- `qrk_analysis`: canonical theory, numerical integration, feasibility checks,
  slack searches, and the integer `smallest_D` API.
- `experiments.heatmaps`: simulation kernels, theoretical model selection,
  multiprocessing generation, and stable text output.
- `qrk_adv`: deprecated compatibility imports. It contains no numerical
  implementation and will be removed after callers migrate.
- `heatmap_data_generation`: compatibility facade for historical experiment
  imports.
- `legacy`: archived exploratory scripts; do not build new work on them.

## Main API

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

`smallest_D` returns an integer and the flat diagnostic fields
`alpha_0`, `alpha_prime`, `c`, `p_l_c`, `p_u`, `failure_prob`, and
`hit_ceiling`. The compatibility parameter `D_precision` is accepted but does
not affect the integer bisection.

## Figures and experiments

Regenerate the paper-bound figures from this repository root:

```bash
python -m qrk_analysis.programs.demo_paper_bounds --paper --recompute
python -m qrk_analysis.programs.demo_paper_bounds --extra --recompute
```

The paper command writes formal PDFs to the parent paper's
`PR_quantile/figures/` directory and caches curves under
`figure/paper_bounds/cache/`. Gaussian exploratory output remains under
`figure/paper_extra/`.

The existing heatmap drivers remain available:

```bash
python heatmap_generation_D_vs_T_demo.py
python heatmap_generation_D_vs_beta_demo.py
```

These experiments can be expensive. Reduce horizons, grids, sample counts,
and worker counts for smoke tests.
