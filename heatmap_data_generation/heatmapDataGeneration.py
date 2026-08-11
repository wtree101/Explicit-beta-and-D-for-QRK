from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qrk_adv.upper_bound import smallest_D
from qrk_adv.search import find_alpha_pair
from functools import partial
from qrk_adv.feasibility import check_feasibility_conditions_random_sup_revised
from qrk_adv.feasibility import check_feasibility_conditions_C_sup_revised
import os


def make_feasibility_check(corruption_type, c_min, c_max):
    match corruption_type:
        case "adversarial":
            return None
        case "sup_c":
            return partial(check_feasibility_conditions_C_sup_revised,num_grid_Q=2,C_min=c_min,C_max=c_max,num_points_C=20)
        case "sup_rand":
            return partial(check_feasibility_conditions_random_sup_revised,num_grid_Q=20,num_points_C=50)
        case _:
            raise ValueError(f"Unknown corruption_type: {corruption_type}")


def c_from_feasibility_result(result):
    if result is None:
        return np.nan
    return result.get("c", result.get("c_min", np.nan))


def cell_bound_c(T, beta, D, q, delta_f, c_target, feasibility_check, maximize_c):
    alpha_pair, result = find_alpha_pair(
        T,
        beta,
        D,
        q,
        delta_f,
        c_target=c_target,
        feasibility_check=feasibility_check,
        maximize_c=maximize_c,
    )
    if alpha_pair is None:
        return np.nan
    return c_from_feasibility_result(result)


def c_at_interval(c, interval_index):
    c_array = np.asarray(c)
    if c_array.ndim == 0:
        return float(c_array)
    return float(c_array[interval_index])


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
    s_max
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
    s_max
):
    xk = np.zeros(np.shape(x))
    for i in range(T_max):
        xk = streaming_subsampled_qRK_step(x,xk,q,beta,D,corruption_type,c_min,c_max,s_min,s_max)[0]

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
    s_max
):
    # Returns boolean-valued array with k-th place corresponding to k*T_intervals iteration succeeding
    errs = np.zeros(int(T_max/T_intervals))
    xk = np.zeros(np.shape(x))

    for i in range(T_max):
        if i % T_intervals == 0:
            c_i = c_at_interval(c, int(i/T_intervals))
            # Squared Relative Err < (1-c/n)^T
            errs[int(i/T_intervals)] = np.linalg.norm(xk-x)**2 / (np.linalg.norm(x)**2) <= (1-c_i/n)**i
        (xk,q_e) = streaming_subsampled_qRK_step(x,xk,q,beta,D,corruption_type,c_min,c_max,s_min,s_max)
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
    c_success_mode="fixed",
    maximize_c_for_success=True
):
    # Min D parameters
    delta_f = 0.1
    D_max = 500
    feasibility_check = make_feasibility_check(corruption_type, c_min, c_max)

    match D_vs_TYPE:
        case "D_vs_T":
            open("./q_e/most_recent_q_e.txt","w").close()

            pool = Pool(processes=48)

            sample_success = np.zeros((num_samples,int(T_max/T_intervals)))
            mean_success = np.zeros((len(D_sample_sizes),int(T_max/T_intervals)))

            # Run samples and log success rate for each (D,T) pair
            D_pos = 0
            c_bound_values = np.full((len(D_sample_sizes),int(T_max/T_intervals)),np.nan)
            for D in tqdm(D_sample_sizes):
                c_for_success = c
                if c_success_mode == "bound":
                    c_for_success = np.array([
                        c if i == 0 else cell_bound_c(i, beta, D, q, delta_f, c, feasibility_check, maximize_c_for_success)
                        for i in range(0, T_max, T_intervals)
                    ])
                    c_bound_values[D_pos,:] = c_for_success
                elif c_success_mode != "fixed":
                    raise ValueError(f"Unknown c_success_mode: {c_success_mode}")

                sample_results = pool.starmap(run_qRK_subsample_D_vs_T,[(D,T_max,T_intervals,x,q,beta,n,c_for_success,corruption_type,c_min,c_max,s_min,s_max)]*num_samples)
                sample_success = np.array([r[0] for r in sample_results])
                sample_q_e = np.array([r[1] for r in sample_results])
                mean_success[D_pos,:] = np.mean(sample_success,axis=0)

                with open("./q_e/most_recent_q_e.txt","a") as f:
                    f.write(f"(D:{D}) {np.mean(sample_q_e)}\n")

                D_pos += 1

            save_heat_map_matrix(D_vs_TYPE="D_vs_T",data_type="",mean_success=np.matrix(mean_success),n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta=beta,T_intervals=T_intervals,c_success_mode=c_success_mode,maximize_c_for_success=maximize_c_for_success)
            if c_success_mode == "bound":
                save_heat_map_matrix(D_vs_TYPE="D_vs_T",data_type="c_bound",mean_success=np.matrix(c_bound_values),n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta=beta,T_intervals=T_intervals,c_success_mode=c_success_mode,maximize_c_for_success=maximize_c_for_success)

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
            \tc_success_mode:\t{c_success_mode}
            \tmaximize_c_for_success:\t{maximize_c_for_success}
            \tD_min:\t\t\t{np.min(D_sample_sizes)}
            \tD_max:\t\t\t{np.max(D_sample_sizes)}
            \tnum_samples:\t\t{num_samples}
            \tT_intervals:\t\t{T_intervals}
            \tT_max:\t\t\t{T_max}
            \tcorruption_type:\t{corruption_type}""")
            
        case "D_vs_beta":
            pool = Pool(processes=48)

            sample_success = np.zeros(num_samples)
            mean_success = np.zeros((len(D_sample_sizes)*len(beta_samples)))

            # Run samples and log success rate for each (D,beta) pair
            c_bound_values = np.full((len(D_sample_sizes),len(beta_samples)),np.nan)
            pos = 0
            for D_pos, D in tqdm(list(enumerate(D_sample_sizes))):
                for beta_pos, beta in enumerate(beta_samples):
                    c_for_success = c
                    if c_success_mode == "bound":
                        c_for_success = cell_bound_c(T_max, beta, D, q, delta_f, c, feasibility_check, maximize_c_for_success)
                        c_bound_values[D_pos,beta_pos] = c_for_success
                    elif c_success_mode != "fixed":
                        raise ValueError(f"Unknown c_success_mode: {c_success_mode}")

                    sample_success = np.array(pool.starmap(run_qRK_subsample_D_vs_beta,[(D,T_max,x,q,beta,n,c_for_success,corruption_type,c_min,c_max,s_min,s_max)]*num_samples))
                    mean_success[pos] = np.mean(sample_success)
                    pos += 1
            
            mean_success =  np.reshape(mean_success,(len(D_sample_sizes),len(beta_samples)))
            save_heat_map_matrix(D_vs_TYPE="D_vs_beta",data_type="",mean_success=mean_success,n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta_samples=beta_samples,c_success_mode=c_success_mode,maximize_c_for_success=maximize_c_for_success)
            if c_success_mode == "bound":
                save_heat_map_matrix(D_vs_TYPE="D_vs_beta",data_type="c_bound",mean_success=c_bound_values,n=n,D_sample_sizes=D_sample_sizes,num_samples=num_samples,T_max=T_max,q=q,c=c,corruption_type=corruption_type,beta_samples=beta_samples,c_success_mode=c_success_mode,maximize_c_for_success=maximize_c_for_success)

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
            \tc_success_mode:\t{c_success_mode}
            \tmaximize_c_for_success:\t{maximize_c_for_success}
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
    c_success_mode="fixed",
    maximize_c_for_success=True
):
    # Save success matrix
    mean_success_mat = np.matrix(mean_success)
    
    fullfilename = ""
    filepath = "./heat_map_raw_data/"
    c_success_suffix = "" if c_success_mode == "fixed" else f"__c_success={c_success_mode}__max_c={maximize_c_for_success}"
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
                case "c_bound":
                    filename2 = "__c_bound"
                case "":
                    filename2 = ""
            filename3 = f"__n={n}__q={q*100:2.0f}__beta_min={np.min(beta_samples)*100:.0f}__beta_max={np.max(beta_samples)*100:.0f}__D_min={np.min(D_sample_sizes)}__D_max={np.max(D_sample_sizes)}__c={c:1.0e}__num_samples={num_samples}__T_max={T_max}__corruption_type={corruption_type}{c_success_suffix}.txt"
            fullfilename = filepath + filename1 + filename2 + filename3

        case "D_vs_T":
            filename1 = "D_vs_T"
            match data_type:
                case "D_min":
                    filename2 = "__D_MIN"
                case "c_bound":
                    filename2 = "__c_bound"
                case "":
                    filename2 = ""
            filename3 = f"__n={n}__q={q*100:2.0f}__beta={beta*100:.0f}__D_min={np.min(D_sample_sizes)}__D_max={np.max(D_sample_sizes)}__c={c:1.0e}__num_samples={num_samples}__T_intervals={T_intervals}__T_max={T_max}__corruption_type={corruption_type}{c_success_suffix}.txt"    
            # fullfilename = os.path.join(os.pardir, filepath + filename1 + filename2 + filename3)
            fullfilename = filepath + filename1 + filename2 + filename3
            print(fullfilename)

    with open(fullfilename,'wb') as f:
        for line in mean_success_mat:
            np.savetxt(f, line, fmt='%.8f')

    if data_type == "":
        print(f"Filename: {fullfilename}\n")
