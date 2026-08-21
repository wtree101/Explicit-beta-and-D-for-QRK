from experiments.heatmaps import generate_heat_map_matrix
import numpy as np

if __name__ == '__main__':
    # Streaming system parameters
    n = 100

    # Noise Parameters
    #   c_min / c_max for epsilon ~ Unif[c_min , c_max]
    #   s_min / s_max for epsilon ~ Normal(0,s) for s ~ Unif[s_min , s_max]
    c_min = -1000
    c_max = 1000
    s_min = 0
    s_max = 10
    feasibility_C_min = 0
    feasibility_C_max = 20
    #   Corruption types: sup_c / sup_rand / adversarial
    corruption_type = "adversarial"

    # True solution
    x = np.random.normal(size=n)
    x = x / np.linalg.norm(x)
    num_workers = 8
    random_seed = 20260814
    # Algorithm parameters
    q = 0.8
    beta_samples = (np.arange(21))*0.001
    D_sample_sizes = (np.arange(60))*2 + 2
    T_max = 10_000

    # Sampling Parameters
    num_samples = 100

    # Empirical success criterion and theoretical contraction target.
    c_success = 0.01
    c_theory = 0.01

    generate_heat_map_matrix(
        D_vs_TYPE="D_vs_beta",
        D_sample_sizes=D_sample_sizes,
        num_samples=num_samples,
        T_max=T_max,
        x=x,
        q=q,
        n=n,
        c_success=c_success,
        c_theory=c_theory,
        corruption_type=corruption_type,
        beta_samples=beta_samples,
        c_min=c_min,
        c_max=c_max,
        s_min=s_min,
        s_max=s_max,
        feasibility_C_min=feasibility_C_min,
        feasibility_C_max=feasibility_C_max,
    )
