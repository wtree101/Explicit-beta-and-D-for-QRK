# Intended paper heatmap settings

This file records the target settings for formal heatmap experiments. Demo
scripts may temporarily use smaller values for exploratory runs; align them
with this record before regenerating paper data.

## Common parameters

```python
n = 100
q = 0.8
beta = 0.01
D_sample_sizes = np.arange(30) + 1
T_intervals = 100
T_max = 20_000

num_samples = 100
num_workers = 8
random_seed = 2026
c_success = 0.05
c_theory = 0.05
```

For the current paper heatmaps, both values are `0.01` under Massart noise and
`0.05` under oblivious noise. They are separate parameters and need not remain
equal in future experiments.

The normalized true solution is generated reproducibly:

```python
rng = np.random.default_rng(random_seed)
x = rng.normal(size=n)
x /= np.linalg.norm(x)
```

## Noise ranges

```python
# Fixed oblivious noise used by corruption_type="sup_c".
c_min = -1000
c_max = 1000

# Gaussian exploration uses variance sampled from this interval.
s_min = 0
s_max = 10

# Independent large-noise simulation ranges.
quantile_noise_min = -2000
quantile_noise_max = 2000
update_noise_min = -1000
update_noise_max = 1000

# Numerical fixed-C range for the theoretical feasibility check.
feasibility_C_min = 0
feasibility_C_max = 20
```

## Comparisons

Use the common parameters for both formal runs and change only the corruption
model:

```python
corruption_type = "sup_c"       # fixed oblivious corruption
corruption_type = "adversarial" # Massart/adversarial corruption
```

For `adversarial`, the simulated fixed-noise ranges are unused. Keeping the
remaining parameters aligned makes the two heatmaps directly comparable.
