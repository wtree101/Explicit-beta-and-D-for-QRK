import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from qrk_adv.upper_bound import smallest_D
from functools import partial
from qrk_adv.feasibility import check_feasibility_conditions_random_sup_revised
from qrk_adv.feasibility import check_feasibility_conditions_C_sup_revised


def make_feasibility_check(
    corruption_type,
    feasibility_C_min=0.0,
    feasibility_C_max=100.0,
):
    match corruption_type:
        case "adversarial":
            return None
        case "sup_c" | "oblivious_large":
            return partial(
                check_feasibility_conditions_C_sup_revised,
                num_grid_Q=2,
                C_min=feasibility_C_min,
                C_max=feasibility_C_max,
                num_points_C=20,
            )
        case "sup_rand":
            return partial(check_feasibility_conditions_random_sup_revised,num_grid_Q=20,num_points_C=50)
        case _:
            raise ValueError(f"Unknown corruption_type: {corruption_type}")


def validate_oblivious_large_noise(
    quantile_noise_min,
    quantile_noise_max,
    update_noise_min,
    update_noise_max,
):
    values = (
        quantile_noise_min,
        quantile_noise_max,
        update_noise_min,
        update_noise_max,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("oblivious_large noise values must be finite")
    if quantile_noise_min > quantile_noise_max:
        raise ValueError(
            "oblivious_large requires quantile_noise_min <= quantile_noise_max"
        )
    if update_noise_min > update_noise_max:
        raise ValueError(
            "oblivious_large requires update_noise_min <= update_noise_max"
        )


def streaming_subsampled_qRK_step(
    x,
    xk,
    q,
    beta,
    D,
    corruption_type,
    c_min,
    c_max,
    s_min,
    s_max,
    *,
    quantile_noise_min=1e16,
    quantile_noise_max=1e16,
    update_noise_min=1e8,
    update_noise_max=1e8,
):
    n = len(x)
    A = np.random.normal(size=(D+1,n))
    row_norms = (np.linalg.norm(A,axis=1))
    A = A / row_norms[:,np.newaxis]

    match corruption_type:
        case "sup_c" | "sup_rand":
            if corruption_type == "sup_c":
                epsilon = np.random.uniform(low=c_min,high=c_max,size=(D+1,))
            else:
                epsilon = np.array([np.random.normal(scale=np.sqrt(s)) for s in np.random.uniform(low=s_min,high=s_max,size=(D+1,))])
            xi = np.random.binomial(1,beta,size=(D+1,))
            residuals = A @ (x - xk) + xi * epsilon

            Q = np.quantile(abs(residuals[1:]),q)
            q_e = np.mean(abs(residuals[1:])<=Q)
            if abs(residuals[0]) <= np.quantile(abs(residuals[1:]),q):
                xk = xk + residuals[0] * A[0,:].T
        case "oblivious_large":
            validate_oblivious_large_noise(
                quantile_noise_min,
                quantile_noise_max,
                update_noise_min,
                update_noise_max,
            )
            xi = np.random.binomial(1, beta, size=(D + 1,))
            clean_residuals = A @ (x - xk)
            quantile_noise = np.random.uniform(
                low=quantile_noise_min,
                high=quantile_noise_max,
                size=D,
            )
            update_noise = np.random.uniform(
                low=update_noise_min,
                high=update_noise_max,
            )
            quantile_residuals = (
                clean_residuals[1:] + xi[1:] * quantile_noise
            )
            update_residual = clean_residuals[0] + xi[0] * update_noise
            Q = np.quantile(abs(quantile_residuals), q)
            q_e = np.mean(abs(quantile_residuals) <= Q)
            if abs(update_residual) <= Q:
                xk = xk + update_residual * A[0, :].T
        case "adversarial":
            xi = np.random.binomial(1,beta,size=(D+1,))
            # epsilon = np.random.normal(size=(D+1,))
            
            if xi[0] == 1:
                # Corrupt sample
                epsilon = np.ones((D+1,)) * 1e16
                test_residuals = (A @ (x - xk)) * (1 - xi) + epsilon * xi # if not corrupt, no change. if corrupt entry, set to +1e16
                Q = np.quantile(abs(test_residuals[1:]),q)
                xk = xk + np.sign(A[0,:] @ (xk - x)) * Q * A[0,:].T
            else:
                # Not corrupt sample
                epsilon = A @ (xk - x)
                test_residuals = A @ (x - xk) + epsilon * xi
                Q = np.quantile(abs(test_residuals[1:]),q)
                if abs(A[0,:] @ (x - xk)) <= Q:
                    xk = xk + A[0,:] @ (x - xk) * A[0,:].T
                    
            q_e = np.mean(abs(test_residuals[1:])<=Q)

    return (xk,q_e)

def run_qRK_subsample_D_vs_beta(
    D,
    T_max,
    x,
    q,
    beta,
    n,
    c,
    corruption_type,
    c_min,
    c_max,
    s_min,
    s_max,
    quantile_noise_min=1e16,
    quantile_noise_max=1e16,
    update_noise_min=1e8,
    update_noise_max=1e8,
    random_seed=None,
):
    if random_seed is not None:
        np.random.seed(random_seed)

    xk = np.zeros(np.shape(x))
    for i in range(T_max):
        xk = streaming_subsampled_qRK_step(
            x, xk, q, beta, D, corruption_type, c_min, c_max, s_min, s_max,
            quantile_noise_min=quantile_noise_min,
            quantile_noise_max=quantile_noise_max,
            update_noise_min=update_noise_min,
            update_noise_max=update_noise_max,
        )[0]

    # Squared Relative Err < (1-c/n)^T
    return np.linalg.norm(xk-x)**2 / (np.linalg.norm(x)**2) < (1-c/n)**T_max

def run_qRK_subsample_D_vs_T(
    D,
    T_max,
    T_intervals,
    x,
    q,
    beta,
    n,
    c,
    corruption_type,
    c_min,
    c_max,
    s_min,
    s_max,
    quantile_noise_min=1e16,
    quantile_noise_max=1e16,
    update_noise_min=1e8,
    update_noise_max=1e8,
    random_seed=None,
):
    if random_seed is not None:
        np.random.seed(random_seed)

    # Returns boolean-valued array with k-th place corresponding to k*T_intervals iteration succeeding
    errs = np.zeros(int(T_max/T_intervals))
    xk = np.zeros(np.shape(x))

    for i in range(T_max):
        (xk,q_e) = streaming_subsampled_qRK_step(
            x, xk, q, beta, D, corruption_type, c_min, c_max, s_min, s_max,
            quantile_noise_min=quantile_noise_min,
            quantile_noise_max=quantile_noise_max,
            update_noise_min=update_noise_min,
            update_noise_max=update_noise_max,
        )
        completed_iterations = i + 1
        if completed_iterations % T_intervals == 0:
            interval_index = int(completed_iterations/T_intervals) - 1
            # Squared Relative Err < (1-c/n)^T
            errs[interval_index] = np.linalg.norm(xk-x)**2 / (np.linalg.norm(x)**2) <= (1-c/n)**completed_iterations
    return (errs,q_e)

def generate_heat_map_matrix(
    D_vs_TYPE,
    D_sample_sizes,
    num_samples,
    T_max,
    x,
    q,
    n,
    c,
    corruption_type,
    beta=0,
    T_intervals=1,
    beta_samples=np.zeros(1),
    c_min=0,
    c_max=1,
    s_min=0,
    s_max=1,
    quantile_noise_min=1e16,
    quantile_noise_max=1e16,
    update_noise_min=1e8,
    update_noise_max=1e8,
    feasibility_C_min=0.0,
    feasibility_C_max=100.0,
    num_workers=None,
    random_seed=None,
):
    # Min D parameters
    delta_f = 0.1
    D_max = 500
    if corruption_type == "oblivious_large":
        validate_oblivious_large_noise(
            quantile_noise_min,
            quantile_noise_max,
            update_noise_min,
            update_noise_max,
        )
    feasibility_check = make_feasibility_check(
        corruption_type,
        feasibility_C_min=feasibility_C_min,
        feasibility_C_max=feasibility_C_max,
    )
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if num_workers is not None and num_workers < 1:
        raise ValueError("num_workers must be at least 1 or None")

    available_workers = os.cpu_count() or 1
    worker_count = min(num_samples, num_workers or available_workers)
    seed_entropy = np.random.SeedSequence(random_seed).entropy

    def sample_seeds(parameter_index):
        return [
            int(
                np.random.SeedSequence(
                    [seed_entropy, parameter_index, sample_index]
                ).generate_state(1)[0]
            )
            for sample_index in range(num_samples)
        ]

    match D_vs_TYPE:
        case "D_vs_T":
            open("./q_e/most_recent_q_e.txt","w").close()

            pool = Pool(processes=worker_count)

            sample_success = np.zeros((num_samples,int(T_max/T_intervals)))
            mean_success = np.zeros((len(D_sample_sizes),int(T_max/T_intervals)))

            # Run samples and log success rate for each (D,T) pair
            try:
                for D_pos, D in enumerate(tqdm(D_sample_sizes)):
                    sample_results = pool.starmap(
                        run_qRK_subsample_D_vs_T,
                        [
                            (
                                D, T_max, T_intervals, x, q, beta, n, c,
                                corruption_type, c_min, c_max, s_min, s_max,
                                quantile_noise_min, quantile_noise_max,
                                update_noise_min, update_noise_max, seed,
                            )
                            for seed in sample_seeds(D_pos)
                        ],
                    )
                    sample_success = np.array([r[0] for r in sample_results])
                    sample_q_e = np.array([r[1] for r in sample_results])
                    mean_success[D_pos,:] = np.mean(sample_success,axis=0)

                    with open("./q_e/most_recent_q_e.txt","a") as f:
                        f.write(f"(D:{D}) {np.mean(sample_q_e)}\n")
            finally:
                pool.close()
                pool.join()

            save_heat_map_matrix(D_vs_TYPE="D_vs_T",data_type="",mean_success=np.matrix(mean_success),n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta=beta,T_intervals=T_intervals)

            # Generate and save D_min values
            D_min_vals = np.zeros(int(T_max/T_intervals))
            for i in tqdm(range(int(T_max/T_intervals))):
                D_min_vals[i] = smallest_D(beta, (i+1)*T_intervals, q, D_max=D_max, delta_f=delta_f, c_target=c, feasibility_check=feasibility_check)["smallest_D"]
            save_heat_map_matrix(D_vs_TYPE="D_vs_T",data_type="D_min",mean_success=np.matrix(D_min_vals).T,n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta=beta,T_intervals=T_intervals)

            print(f"""Variables:
            \tn:\t\t\t{n}
            \tq:\t\t\t{q}
            \tbeta:\t\t\t{beta}
            \tc:\t\t\t{c}
            \tD_min:\t\t\t{np.min(D_sample_sizes)}
            \tD_max:\t\t\t{np.max(D_sample_sizes)}
            \tnum_samples:\t\t{num_samples}
            \tT_intervals:\t\t{T_intervals}
            \tT_max:\t\t\t{T_max}
            \tcorruption_type:\t{corruption_type}""")
            
        case "D_vs_beta":
            pool = Pool(processes=worker_count)

            sample_success = np.zeros(num_samples)
            mean_success = np.zeros((len(D_sample_sizes)*len(beta_samples)))

            # Run samples and log success rate for each (D,beta) pair
            pos = 0
            try:
                for D in tqdm(D_sample_sizes):
                    for beta in beta_samples:
                        sample_success = np.array(
                            pool.starmap(
                                run_qRK_subsample_D_vs_beta,
                                [
                                    (
                                        D, T_max, x, q, beta, n, c,
                                        corruption_type, c_min, c_max, s_min,
                                        s_max, quantile_noise_min,
                                        quantile_noise_max, update_noise_min,
                                        update_noise_max, seed,
                                    )
                                    for seed in sample_seeds(pos)
                                ],
                            )
                        )
                        mean_success[pos] = np.mean(sample_success)
                        pos += 1
            finally:
                pool.close()
                pool.join()
            
            mean_success =  np.reshape(mean_success,(len(D_sample_sizes),len(beta_samples)))
            save_heat_map_matrix(D_vs_TYPE="D_vs_beta",data_type="",mean_success=mean_success,n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta_samples=beta_samples)

            D_min_vals = np.zeros(len(beta_samples))
            for i in tqdm(range(len(beta_samples))):
                D_min_vals[i] = smallest_D(beta_samples[i], T_max, q, D_max=D_max, delta_f=delta_f, c_target=c, feasibility_check=feasibility_check)["smallest_D"]
            save_heat_map_matrix(D_vs_TYPE="D_vs_beta",data_type="D_min",mean_success=np.matrix(D_min_vals),n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta_samples=beta_samples)
            save_heat_map_matrix(D_vs_TYPE="D_vs_beta",data_type="D_samples",mean_success=np.matrix(D_sample_sizes).T,n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta_samples=beta_samples)
            save_heat_map_matrix(D_vs_TYPE="D_vs_beta",data_type="beta_samples",mean_success=np.matrix(beta_samples).T,n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta_samples=beta_samples)

            print(np.array2string(np.matrix(D_min_vals)))

            print(f"""Variables:
            \tn:\t\t\t{n}
            \tq:\t\t\t{q}
            \tbeta_min:\t\t{np.min(beta_samples)}
            \tbeta_max:\t\t{np.max(beta_samples)}
            \tc:\t\t\t{c}
            \tD_min:\t\t\t{np.min(D_sample_sizes)}
            \tD_max:\t\t\t{np.max(D_sample_sizes)}
            \tnum_samples:\t\t{num_samples}
            \tT_max:\t\t\t{T_max}
            \tcorruption_type:\t{corruption_type}""")

def save_heat_map_matrix(
    D_vs_TYPE,
    data_type,
    mean_success,
    n,
    D_sample_sizes,
    num_samples,
    T_max,
    q,
    c,
    corruption_type,
    beta=0,
    T_intervals=1,
    beta_samples=np.zeros(1),
):
    # Save success matrix
    mean_success_mat = np.matrix(mean_success)
    
    fullfilename = ""
    filepath = "./heat_map_raw_data/"
    match D_vs_TYPE:
        case "D_vs_beta":
            filename1 = "D_vs_beta"
            match data_type:
                case "D_min":
                    filename2 = "__D_min"
                case "D_samples":
                    filename2 = "__D_samples"
                case "beta_samples":
                    filename2 = "__beta_samples"
                case "":
                    filename2 = ""
            filename3 = f"__n={n}__q={q*100:2.0f}__beta_min={np.min(beta_samples)*100:.0f}__beta_max={np.max(beta_samples)*100:.0f}__D_min={np.min(D_sample_sizes)}__D_max={np.max(D_sample_sizes)}__c={c:1.0e}__num_samples={num_samples}__T_max={T_max}__corruption_type={corruption_type}.txt"
            fullfilename = filepath + filename1 + filename2 + filename3

        case "D_vs_T":
            filename1 = "D_vs_T"
            match data_type:
                case "D_min":
                    filename2 = "__D_MIN"
                case "":
                    filename2 = ""
            filename3 = f"__n={n}__q={q*100:2.0f}__beta={beta*100:.0f}__D_min={np.min(D_sample_sizes)}__D_max={np.max(D_sample_sizes)}__c={c:1.0e}__num_samples={num_samples}__T_intervals={T_intervals}__T_max={T_max}__corruption_type={corruption_type}.txt"
            fullfilename = filepath + filename1 + filename2 + filename3
            print(fullfilename)

    with open(fullfilename,'wb') as f:
        for line in mean_success_mat:
            np.savetxt(f, line, fmt='%.8f')

    if data_type == "":
        print(f"Filename: {fullfilename}\n")
