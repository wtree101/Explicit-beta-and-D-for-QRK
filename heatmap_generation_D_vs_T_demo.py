from experiments.heatmaps import generate_heat_map_matrix
import numpy as np

if __name__ == '__main__':
    # Streaming system parameters
    n = 100

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
    T_intervals = 1000
    T_max = 10_000

    # Sampling Parameters
    num_samples = 10
    # None uses min(num_samples, available CPU cores).
    num_workers = 8
    corruption_type = "sup_c"  # "sup_c" / "oblivious_large" / "adversarial"

    # Empirical success criterion and theoretical contraction target.
    c_success = 0.05
    c_theory = 0.05

    generate_heat_map_matrix(
        D_vs_TYPE="D_vs_T",
        D_sample_sizes=D_sample_sizes,
        num_samples=num_samples,
        T_max=T_max,
        x=x,
        q=q,
        n=n,
        c_success=c_success,
        c_theory=c_theory,
        corruption_type=corruption_type,
        beta=beta,
        T_intervals=T_intervals,
        c_min=c_min,
        c_max=c_max,
        s_min=s_min,
        s_max=s_max,
        quantile_noise_min=quantile_noise_min,
        quantile_noise_max=quantile_noise_max,
        update_noise_min=update_noise_min,
        update_noise_max=update_noise_max,
        feasibility_C_min=feasibility_C_min,
        feasibility_C_max=feasibility_C_max,
        num_workers=num_workers,
        random_seed=random_seed,
    )
