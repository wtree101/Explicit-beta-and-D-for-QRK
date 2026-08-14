# Repository Guidelines

## Project purpose

This repository is a small research codebase for studying upper bounds on the
QRK quantile-subsample size `D`. It contains numerical feasibility checks for
adversarial Massart noise, fixed-magnitude oblivious noise, and Gaussian
oblivious noise, plus Monte Carlo experiments that compare the theoretical
bound with empirical convergence.

The repository is run directly from its root; it is not an installed package
and has no `pyproject.toml` or `setup.py`.

## Repository map

- `qrk_adv/`: reusable theory and search code.
  - `divergence.py`: Bernoulli KL divergence.
  - `quantile.py`: half-normal quantiles and conditional second moments.
  - `noise.py`: fixed-`C` and Gaussian error-increase calculations.
  - `feasibility.py`: adversarial, fixed-`C`, and Gaussian feasibility checks.
  - `search.py`: searches for feasible slack parameters at a fixed `D`.
  - `upper_bound.py`: `smallest_D`, the primary public entry point.
  - `debug.py`: opt-in diagnostic logging controlled by `set_debug`.
- `heatmap_data_generation/heatmapDataGeneration.py`: streaming QRK simulation,
  multiprocessing sweeps, theoretical `D_min` calculation, and text output.
- `tests/`: fast `unittest` regression tests for feasibility behavior, heatmap
  success criteria, multiprocessing setup, and `oblivious_large` noise.
- `demo.py`: small theoretical-bound example; its active `__main__` path runs a
  single adversarial feasibility calculation.
- `heatmap_generation_D_vs_T_demo.py` and
  `heatmap_generation_D_vs_beta_demo.py`: expensive Monte Carlo heatmap drivers.
- `convergence_curves_D_demo.py`: fixed-horizon convergence experiment; writes
  PNG and NPZ outputs under `figure/`.
- `heatmap_data_display/`: notebooks and MATLAB scripts for displaying saved
  heatmap data. `archive/` contains historical display material only.
- Top-level notebooks are exploratory analyses, not part of the automated test
  suite.
- `heat_map_raw_data/`, `q_e/`, and `figure/`: generated experiment outputs.

## Environment and dependencies

Always run Python code and tests in the Conda environment `base`:

```bash
conda activate base
python --version
```

In a non-interactive shell where `conda activate` is not initialized, use:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
```

Install the unpinned scientific dependencies, if needed, with:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root so imports such as `qrk_adv` and
`heatmap_data_generation` resolve correctly. The code uses Python 3.10 syntax
(`match`/`case`), and the current `base` environment uses Python 3.10.

## Testing

The canonical fast regression command is:

```bash
python -m unittest discover -s tests -v
```

To run one module or one test:

```bash
python -m unittest tests.test_fixed_success -v
python -m unittest tests.test_oblivious_large.ObliviousLargeNoiseTests.test_clean_quantile_sample_rejects_corrupted_update -v
```

There is no configured linter, formatter, or type checker. At minimum, run the
full unit suite after changing reusable Python code. Add or update focused
`unittest` coverage when changing search behavior, feasibility result fields,
noise generation, saved matrix shapes/names, worker selection, or success
criteria. Keep tests small and deterministic by using seeds, mocks, and
temporary directories; do not run full Monte Carlo sweeps as unit tests.

## Core APIs and invariants

- `qrk_adv.upper_bound.smallest_D(...)` binary-searches `D` and delegates the
  slack search to `find_alpha_pair`. Preserve its result contract, especially
  `smallest_D`, `hit_ceiling`, `alpha_0`, `alpha_prime`, `c`, and
  `failure_prob`. An infeasible bounded search returns `smallest_D=None`.
- Feasibility checks accept the common positional arguments
  `(T, beta, D, q, alpha_0, alpha_prime, delta_f)`, accept `c_target`, and accept
  keyword-only `enforce_failure_probability`. They return a dict containing
  `feasible`, `failure_prob`, and either `c` or `c_min`.
- `smallest_D` normalizes a revised check's `c_min` to the public result key
  `c`. Code consuming a feasibility check directly must still handle `c_min`.
- Valid inputs require `0 <= beta < min(q, 1-q)`. Keep the documented bounds on
  `alpha_0` and `alpha_prime` and the distinction between contraction failure
  and failure-probability failure.
- `find_max_c_without_failure_constraint` is exploratory only. Its contraction
  must not replace the fixed empirical heatmap success criterion or be reported
  as a high-probability theoretical bound.
- Numerical grid sizes and ranges in the fixed-`C` and Gaussian checks are a
  deliberate accuracy/runtime tradeoff. Do not silently reduce or reinterpret
  them.

## Simulation conventions

Keep the supported `corruption_type` values and their meanings aligned across
the generator, demos, README, and tests:

- `adversarial`: adversarial Massart-style update.
- `sup_c`: simulation noise sampled uniformly from `[c_min, c_max]`; theory
  uses the fixed-`C` supremum check.
- `sup_rand`: Gaussian oblivious simulation noise; theory uses the supremum
  over sigma.
- `oblivious_large`: separate finite uniform noise intervals for quantile rows
  and the candidate update; theory uses the fixed-`C` supremum check.

Do not introduce `fixed` as a corruption type without adding it consistently
as an alias. For `oblivious_large`, keep quantile and update noise draws
independent, validate ordered finite endpoints, and never replace the finite
defaults with infinity: infinite values can contaminate quantiles or updates
with `NaN`/`inf`.

The simulation intervals are intentionally independent of
`feasibility_C_min`/`feasibility_C_max`, which control only the numerical
fixed-`C` theoretical supremum. Do not infer one range from the other.

Heatmap success is defined uniformly by
`squared_relative_error <= (1 - c / n) ** T` using the preset `c`. The separately
computed `D_min` curve uses that same value as the theoretical `c_target`. Do
not restore the removed `c_bound`/`c_success_mode` behavior or filename suffixes.

## Experiment and output hygiene

The heatmap and convergence drivers can be CPU-intensive and may use
multiprocessing. Before running one, inspect and usually reduce `T_max`, sample
counts, grid sizes, `D_list`, and worker counts for a smoke test. Seed new
stochastic regression scenarios when reproducibility matters. Preserve the
current worker cap: no more workers than samples.

Experiment scripts write into existing output directories:

- heatmap matrices: `heat_map_raw_data/*.txt`
- quantile diagnostics: `q_e/most_recent_q_e.txt`
- plots and convergence arrays: `figure/*.png` and `figure/*.npz`

Treat existing notebooks, data files, figures, cache files, and worktree edits
as user-owned. Do not delete, regenerate, or commit large derived outputs unless
the task explicitly requires it. When a change affects output naming or matrix
orientation, update the Python writer, display notebooks/MATLAB scripts,
README, and tests together.

## Coding style

- Follow the surrounding NumPy/SciPy style and use four-space indentation.
- Prefer descriptive snake_case names and short, testable functions.
- Use NumPy arrays for new numerical code. `np.matrix` remains in the legacy
  text writer for compatibility, but new code should not spread its use.
- Use `numpy.random.default_rng` and explicit seeds for new experiment-level
  code when practical. Existing worker functions use legacy global NumPy RNG;
  preserve their per-sample seeding unless refactoring the whole call path.
- Keep plotting headless in scripts by selecting the `Agg` backend before
  importing `matplotlib.pyplot`.
- Use `qrk_adv.debug.debug_log` for optional library diagnostics rather than
  unconditional prints. Progress and saved-path messages are acceptable in
  top-level experiment drivers.
- Document any public parameter or returned field whose numerical meaning is
  not obvious, particularly whether it belongs to simulation or theory.

## Documentation expectations

Update `README.md` when changing the main API, feasibility-check signatures,
noise-model semantics, empirical success criterion, or user-facing experiment
commands. Comments and docs must distinguish preliminary theoretical bounds
from empirical observations and fixed plotting/success references.
