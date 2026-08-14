# Explicit beta and D in QRK

Minimal self-contained code (`adv_D_upper`) for computing the **upper bound on
the subsample size D** in Quantile-based Randomized Kaczmarz (QRK) under
**adversarial Massart noise**, **fixed noise (supremum over C)**, and
**Gaussian oblivious noise (supremum over σ)**.

---

## Background

QRK solves a corrupted linear system Ax* + ε = b where ε is (βm)-sparse.
At each iteration it computes a q-quantile of the residual using a subsample
of D rows.  Our theory (preliminary result) shows that QRK linearly converges
over T iterations with high probability provided

```
D ≥ C · log(T) / log(1/β)
```

for some absolute constant C.  This code numerically finds the **smallest D**
satisfying the full set of feasibility conditions for given (β, T, q, δ_f).

---

## Package layout

```
adv_D_upper/
├── qrk_adv/
│   ├── divergence.py    # DKL(q || p) for Bernoulli distributions
│   ├── quantile.py      # half-normal quantile, sigma_min^2 lower bound
│   ├── noise.py         # fixed/Gaussian noise error increase utilities
│   ├── feasibility.py   # feasibility conditions (adversarial + variants)
│   ├── search.py        # find feasible (alpha_0, alpha') at fixed D
│   └── upper_bound.py   # binary search for smallest_D  ← main entry point
├── demo.py              # single example + sweep plot (adversarial by default)
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
cd adv_D_upper
pip install -r requirements.txt
python demo.py
```

This prints the smallest D for a single example and saves
`figure/demo_D_vs_beta.png` showing D vs beta.

## Demo figure

![Upper bound on D vs beta](figure/demo_D_vs_beta.png)

---

## Main API

```python
from qrk_adv.upper_bound import smallest_D
from functools import partial
from qrk_adv.feasibility import check_feasibility_conditions_random_sup_revised

result = smallest_D(
    beta    = 0.05,    # corruption fraction
    T       = 20_000,  # number of iterations
    q       = 0.75,    # quantile level
    delta_f = 0.1,     # total failure probability budget
    D_max   = 500,     # search ceiling (raise if result hits it)
  c_target = 0.0,
    feasibility_check = None,  # defaults to adversarial check
)

print(result["smallest_D"])   # smallest feasible D
print(result["c"])             # net contraction coefficient c
print(result["failure_prob"])  # actual failure probability
```

To use the Gaussian worst-sigma check with explicit grid sizes:

```python
from qrk_adv.feasibility import check_feasibility_conditions_random_sup_revised

gaussian_check = partial(
  check_feasibility_conditions_random_sup_revised,
  num_grid_Q=20,
  num_points_C=50,
)

result = smallest_D(
  beta=0.05,
  T=20_000,
  q=0.75,
  delta_f=0.1,
  D_max=500,
  c_target=0.0,
  feasibility_check=gaussian_check,
)
```

To use the **fixed-noise supremum-over-C** check (oblivious worst-case magnitude):

```python
from qrk_adv.feasibility import check_feasibility_conditions_C_sup_revised

fixed_C_check = partial(
  check_feasibility_conditions_C_sup_revised,
  num_grid_Q=2,       # Q_{q,k+1} sweep; 1 often suffices if c increases in q
  C_min=0.0,
  C_max=20.0,
  num_points_C=20,    # grid for sup_C error_increased_C_3
)

result = smallest_D(
  beta=0.05,
  T=20_000,
  q=0.75,
  delta_f=0.1,
  D_max=500,
  c_target=0.0,
  feasibility_check=fixed_C_check,
)

# Revised checks return c_min (not c); smallest_D maps it to result["c"].
print(result["c"], result.get("smallest_D"))
```

### Parameters

| Name        | Type  | Meaning |
|-------------|-------|---------|
| `beta`      | float | Corruption fraction beta in (0, q) and (0, 1-q) |
| `T`         | int   | Number of QRK iterations |
| `q`         | float | Quantile level, e.g. 0.75 |
| `delta_f`   | float | Total failure probability budget |
| `D_max`     | int   | Search ceiling on D (default 500) |
| `D_precision`| float | Binary-search stopping tolerance (default 1) |
| `c_target`  | float | Require c ≥ c_target (default 0) |
| `feasibility_check` | callable | Optional check function (defaults to adversarial) |

### Return value

| Key            | Meaning |
|----------------|---------|
| `smallest_D`   | Smallest feasible D (None if infeasible within D_max) |
| `alpha_0`      | Optimal lower slack parameter |
| `alpha_prime`  | Optimal upper slack parameter |
| `c`            | Net contraction at optimum (`c` or `c_min` from the check) |
| `failure_prob` | Actual failure probability 1 - (1-p_u)^T |
| `hit_ceiling`  | True if result == D_max (increase D_max) |

---

## Feasibility conditions for adversarial case

The two conditions verified at each candidate D are:

1. **Net contraction** (c > 0):  
   `p_l_c = 1 - exp(-DKL(q, β+α₀) · D)`
   `Φ = half_normal_quantile(1 - α'/(1-β))`
   `c = (1-β)·p_l_c·σ_min²(α₀/(1-β)) - β·(Φ² + 2Φ·E|Z|) ≥ c_target`  

2. **Failure probability**:  
   `1 - (1 - β·exp(-DKL(1-q, β+α')·D))^T ≤ δ_f`

---

## Feasibility checks in `qrk_adv.feasibility`

| Function | Noise model |
|----------|-------------|
| `check_feasibility` | Adversarial Massart (preliminary) |
| `check_feasibility_conditions_C_sup_revised` | Fixed magnitude C; sup over `[C_min, C_max]` at each conditional quantile |
| `check_feasibility_conditions_random_sup_revised` | Gaussian C ~ N(0, σ²); sup over `[sigma_min, sigma_max]` |

Custom checks used by exploratory diagnostic searches must accept
`(T, beta, D, q, alpha_0, alpha_prime, delta_f, c_target=...,
*, enforce_failure_probability=True)`. They must return a dict containing
`feasible`, `failure_prob`, and either `c` or `c_min`. When
`enforce_failure_probability=False`, the check should still report the failure
probability but must not reject an otherwise valid contraction candidate because
it exceeds `delta_f`.

## Heatmap `corruption_type` values

The heatmap demo scripts use a string-valued `corruption_type` to select the
simulation noise model and the matching feasibility check:

| `corruption_type` | Meaning |
|-------------------|---------|
| `"adversarial"` | Adversarial Massart-style corruption |
| `"oblivious_large"` | Bernoulli corruption with separate large finite simulation noises for the quantile sample and candidate update |
| `"sup_c"` | Simulation noise sampled uniformly from `c_min <= epsilon <= c_max` |
| `"sup_rand"` | Gaussian oblivious noise; supremum over sigma |

So for the fixed-noise case, set:

```python
corruption_type = "sup_c"
```

Do not use `"fixed"` unless the code is extended to recognize it as an alias.

For the fixed-reference convergence experiment, `"oblivious_large"`
independently samples the noise on each corrupted quantile-reference row from
`Uniform(quantile_noise_min, quantile_noise_max)` and independently samples the
corrupted candidate-row noise from
`Uniform(update_noise_min, update_noise_max)`. The defaults
`quantile_noise_min=quantile_noise_max=1e16` and
`update_noise_min=update_noise_max=1e8` reproduce the previous deterministic
large-noise behavior. Either interval can instead be set to `[-1000, 1000]` to
use uniform oblivious noise for that role. All values are finite on purpose:
IEEE infinity could make quantile interpolation or an accepted update produce
`NaN`/`inf`. The corruption indicators remain Bernoulli with rate `beta`.

The simulation noise parameters are separate from the numerical parameters used
to compute the theoretical `D_min`. Both `"sup_c"` and `"oblivious_large"` use
the fixed-C supremum feasibility check with, by default,

```python
feasibility_C_min = 0
feasibility_C_max = 100
```

This interval approximates a supremum over an unknown oblivious noise magnitude;
it is not inferred from `c_min`, `c_max`, or either simulation-noise interval.
The default grid currently uses 20 points, which can be increased in
`make_feasibility_check` when a finer numerical supremum is needed.

## Heatmap success criterion

The heatmap generator uses the same preset `c` for every heatmap cell. A run
is counted as successful when

```text
squared_relative_error <= (1 - c / n)^T
```

This is a uniform and directly comparable experimental threshold across all
cells. The preset `c` is not claimed to estimate the experiment's actual
contraction rate. The former `c_bound` success mode was removed because the
bound-derived value did not accurately represent that empirical rate.

The separately generated `D_min` curve is different: it still uses the full
theoretical feasibility and failure-probability constraints, with the preset
`c` as `c_target`, and retains its theoretical meaning.

`find_max_c_without_failure_constraint` remains available only for historical
or exploratory diagnostics. Its output must not be used as a heatmap success
criterion. Historical `c_bound` display material is kept under
`heatmap_data_display/archive/` and new heatmap runs do not generate those
matrices or filename suffixes.

## Convergence curves at fixed T

`convergence_curves_D_demo.py` compares full convergence trajectories for a
list of D values at one fixed time horizon. Its default configuration uses
`corruption_type="oblivious_large"`, `D=1,...,10`, and 5 independent trials:

```bash
python3 convergence_curves_D_demo.py
```

For every D, the figure shows the mean squared relative error as a solid line,
a 10%-90% trial band, and `(1-c/n)^t` using the preset fixed `c` as a
black dashed reference. This dashed curve is a fixed success reference,
not an estimate of the actual contraction rate. The script saves both a PNG
and the complete trial data as an NPZ file under `figure/`.

Edit `ExperimentConfig` near the top of the script to change `T`, `D_list`,
`num_trials`, `record_every`, `fixed_c`, or the noise model. The full curves
should be used to compare empirical rates across D; the binary success
criterion intentionally remains uniform and simple.

---

## Notes

- The default feasibility check is the preliminary adversarial analysis.
  Pass a custom check via `feasibility_check` (typically `functools.partial`)
  for the fixed-C or Gaussian supremum variants in `qrk_adv.feasibility`.
- Feasibility requires `β < min(q, 1-q)`.  Near this boundary D grows
  rapidly; increase `D_max` as needed.
- The `c_target` parameter can be used to study robustness: a larger
  c_target forces a faster convergence rate but raises the required D.

## Additional demos
- `demo.py` supports a `FEASIBILITY_CHECK` variable and a `VERBOSE` flag
  for printing sweep progress.
