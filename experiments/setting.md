 n = 100

## Experiment parameters
    # Noise Parameters
    # Used only by sup_c simulation: epsilon ~ Uniform(c_min, c_max).
    c_min = -1000
    c_max = 1000
    # variance for Gaussian noise
    s_min = 0
    s_max = 10
    # oblivious_large simulation noise
    quantile_noise_min = -2000
    quantile_noise_max = 2000
    update_noise_min = -1000
    update_noise_max = 1000

    # Numerical sup_C range for the theoretical oblivious-noise feasibility.
    # This is independent of the simulated noise values above.
    feasibility_C_min = 0
    feasibility_C_max = 20

    # Set to an integer for a reproducible experiment, or None for a fresh run.
    random_seed = 2026
    rng = np.random.default_rng(random_seed)

    # True solution
    x = rng.normal(size=n)
    x = x / np.linalg.norm(x)

    # Algorithm parameters
    q = 0.8
    beta = 0.01
    D_sample_sizes = (np.arange(30))+1
    T_intervals = 100
    T_max = 20_000

    # Sampling Parameters
    num_samples = 100
    # None uses min(num_samples, available CPU cores).
    num_workers = 8
    corruption_type = "sup_c"

    # Min D parameters
    c = 0.05
### Experiment 1 oblivious
# Noise Parameters
    # Used only by sup_c simulation: epsilon ~ Uniform(c_min, c_max).
    c_min = -1000
    c_max = 1000


    # Numerical sup_C range for the theoretical oblivious-noise feasibility.
    # This is independent of the simulated noise values above.
    feasibility_C_min = 0
    feasibility_C_max = 20

    # Set to an integer for a reproducible experiment, or None for a fresh run.
    random_seed = 2026
    rng = np.random.default_rng(random_seed)

    # True solution
    x = rng.normal(size=n)
    x = x / np.linalg.norm(x)

    # Algorithm parameters
    q = 0.8
    beta = 0.01
    D_sample_sizes = (np.arange(30))+1
    T_intervals = 100
    T_max = 20_000

    # Sampling Parameters
    num_samples = 100
    # None uses min(num_samples, available CPU cores).
    num_workers = 8
    corruption_type = "sup_c"

    # Min D parameters
    c = 0.05

### Experiment 2 adversarial
# Noise Parameters
    # Used only by sup_c simulation: epsilon ~ Uniform(c_min, c_max).
    c_min = -1000
    c_max = 1000


    # Numerical sup_C range for the theoretical oblivious-noise feasibility.
    # This is independent of the simulated noise values above.
    feasibility_C_min = 0
    feasibility_C_max = 20

    # Set to an integer for a reproducible experiment, or None for a fresh run.
    random_seed = 2026
    rng = np.random.default_rng(random_seed)

    # True solution
    x = rng.normal(size=n)
    x = x / np.linalg.norm(x)

    # Algorithm parameters
    q = 0.8
    beta = 0.01
    D_sample_sizes = (np.arange(30))+1
    T_intervals = 100
    T_max = 20_000

    # Sampling Parameters
    num_samples = 100
    # None uses min(num_samples, available CPU cores).
    num_workers = 8
    corruption_type = "adversarial"

    # Min D parameters
    c = 0.05