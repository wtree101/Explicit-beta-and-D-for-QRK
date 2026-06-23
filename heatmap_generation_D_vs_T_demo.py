from heatmapDataGeneration import generate_heat_map_matrix
import numpy as np

if __name__ == '__main__':
    # Streaming system parameters
    n = 100

    # Noise Parameters
    c_min = -10
    c_max = 10
    s_min = 0
    s_max = 10

    # True solution
    x = np.random.normal(size=n)
    x = x / np.linalg.norm(x)

    # Algorithm parameters
    q = 0.8
    beta = 0.005
    D_sample_sizes = np.arange(30)+1
    T_intervals = 100
    T_max = 20_000

    # Sampling Parameters
    num_samples = 100
    corruption_type = "adversarial"

    # Min D parameters
    c = 0.001

    generate_heat_map_matrix(
        D_vs_TYPE="D_vs_T",
        D_sample_sizes=D_sample_sizes,
        num_samples=num_samples,
        T_max=T_max,
        x=x,
        q=q,
        n=n,
        c=c,
        corruption_type=corruption_type,
        beta=beta,
        T_intervals=T_intervals,
        c_min=c_min,
        c_max=c_max,
        s_min=s_min,
        s_max=s_max,
    )