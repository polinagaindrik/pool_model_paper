#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

optimization_history = []
output_file = 'out/optimization_history1.csv'

# Define the pool model ODEs (general case)
def pool_model(t, x, param, x0, const):
    (alpha, lambd, n_max, ) = param
    (x01, ) = x0
    (x1, ) = x
    return [alpha * (x1 - x01 * np.exp(- lambd * t)) * (1 - x1 / n_max)]

def ode_model_coculture(t, x, param, x0, ode_args):
    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    (x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls_opt, mu_lm_opt,
    pH_ls_min, pH_ls_opt, pH_lm_min, pH_lm_opt,
    omega_ls_exp, omega_lm_exp,
    omegaT_lm_exp, k_T_inhib0, n,
    N_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp,
    q_acid) = param

    mu_ls = mu_ls_opt * (pH - pH_ls_min) / (pH_ls_opt - pH_ls_min)
    mu_lm = mu_lm_opt * (pH - pH_lm_min) / (pH_lm_opt - pH_lm_min)
    #mu_ls = mu_ls_opt**2 * (pH - pH_ls_min)**2
    #mu_lm = mu_lm_opt**2 * (pH - pH_lm_min)**2

    N_t = 10**N_texp
    omega_ls = 10**(-3) * omega_ls_exp
    omega_lm = 10**(-3) * omega_lm_exp
    omegaT_lm = omegaT_lm_exp
    kappa_T = 10**(-5) * kappa_T_0
    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm = 10**np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp])

    k_T_inhib = k_T_inhib0

    #n = 2 
    toxin_death = omegaT_lm * x_lm_sen * T**n / (k_T_inhib**n + T**n) #omegaT_lm * x_lm_sen * T #

    return [
        (mu_ls * R - omega_ls) * x_ls23K,
        (mu_ls * R - omega_ls) * x_lsCTC494,
        (mu_lm * R  - omega_lm) * x_lm_sen - toxin_death,
        (mu_lm * R  - omega_lm) * x_lm_res,
        -(mu_ls / N_t)*R*x_ls23K - (mu_ls / N_t)*R*x_lsCTC494 - (mu_lm / N_t)*R*x_lm_sen - (mu_lm / N_t)*R*x_lm_res,
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K*x_ls23K + kappa_LA_ls23K_2*R*x_ls23K) + (kappa_LA_lsCTC494*x_lsCTC494 + kappa_LA_lsCTC494_2*R*x_lsCTC494) +
        + kappa_LA_lm  * (x_lm_sen+x_lm_res),  # *R but wo R the curves look better
        0.#- q_acid * LA
    ]

    
def observable(t, x):
    n = np.array([x[0]+x[1], x[2]+x[3]])
    obs = np.concatenate((n, x[5:])) # mb add pH
    return obs


def pH_func(t, pH_series):
    # pH_series = [[pH1, t1], [pH2, t2], [pH3, t3], ...] (n_times x 2)
    pH_arr, time_arr = np.array(pH_series).T
    diff = time_arr - t
    return pH_arr[np.argmin(np.abs(diff))]


# The model solution for given ODEs
def model_ODE_solution(model, t, param, x0, const, t0=0., jac=None, jac_spasity=None):
    sol_model = solve_ivp(model, [t0, t[-1]], x0, dense_output=False, method='LSODA', max_step=0.1, t_eval=t, args=(param, x0, const), rtol=1e-6, atol=1e-6, jac=jac)#, lband=const[1], uband=2*const[1])#, jac_spasity=jac_spasity
    return sol_model.y

# Parameter estimation using minimizstion of the negative log-likelihood function (func)
def optimization_func(func, bnds, args=(), workers=1):
    return differential_evolution(func, args=args, tol=1e-6, atol=1e-6, maxiter=10, mutation=(0.3, 1.9), recombination=0.7, popsize=30,
                                  bounds=bnds, init='latinhypercube', disp=True, polish=False, updating='deferred', workers=workers,
                                  strategy='randtobest1bin', callback=_callback_ll) #init='sobol'

def _callback_ll(intermediate_result):
    """Saves the best solution and functoin value at each iteration."""
    optimization_history.append((intermediate_result.x.copy(), intermediate_result.fun.copy()))  # Save a copy of x to avoid overwriting
    with open(output_file, "a") as f:
        output = f"{len(optimization_history)},"
        for p in intermediate_result.x:
            output += f"{p},"
        f.write(output+f"{intermediate_result.fun}\n")

def cost_arithmetic_mean(J_vect):#ll_ngs, ll_maldi, ll_mibi):
    return np.nanmean([np.nanmean(Ji) for Ji in J_vect])

##################################################################################3
def extract_observables_from_df(dfs):
    (df_x,) = dfs
    exps = sorted(list(set([s.split("_")[0] for s in df_x.columns])))
    days_x = sorted(set([float(f.split("_")[-2]) for f in df_x.columns]))
    obs_x = np.zeros((len(exps), np.shape(df_x)[0], len(days_x)))
    for i, exp in enumerate(exps):
        for k, d in enumerate(days_x):
            df0 = df_x.filter(like=exp).filter(like=f"_{int(d):02d}_")
            if np.shape(df0)[-1] != 0.0:
                obs_x[i, :, k] = np.array(df0.T)[0]
            else:
                obs_x[i, :, k] = np.nan * np.ones((np.shape(df_x)[0]))
    return days_x, [obs_x]

def set_initial_vals(x10, temps, n_cl, pH0=6.):
    return np.concatenate((x10, [1., 0., 0., pH0]))

def calculate_model_params(cost_func, calibr_setup):
    output_file = "out/optimization_history1.csv"
    with open(output_file, "w") as f:
        output = "iteration,"
        for i in range(len(calibr_setup["param_bnds"])):
            output += f"p{i},"
        f.write(output + "cost\n")
    data_array = extract_observables_from_df(calibr_setup["dfs"])
    calibr_setup["data_array"] = data_array
    optim_output = optimization_func(
        cost_func,
        calibr_setup["param_bnds"],
        args=(calibr_setup, None),
        workers=calibr_setup["workers"],
    )
    return np.array(optim_output.x), optim_output.fun

def cost(param, calibr_setup, jac_spasity):
    n_cl = calibr_setup["n_cl"]
    exps = calibr_setup["exps"]
    n_exps = len(exps)
    param_ode = param[n_cl*n_exps:]
    param_ode_new = np.copy(param_ode)
    x0_vals = param[:n_cl*n_exps]

    (df_x, ) = calibr_setup["dfs"]
    _, [obs_x] = calibr_setup["data_array"]
    n_cl = calibr_setup['n_cl']  # np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split("_")[0] for s in df_x.columns])))
    # TODO not clear, should just we compare logaritms?
    x_max = obs_x**2
    x_max[x_max == 0.0] = 1.0
    #ll_x = np.zeros(np.shape(obs_x))
    ll_x = np.zeros(np.shape(obs_x[:, :-1]))
    for i, exp in enumerate(exps):
        #if exp != 'LsCTC494' and exp != 'LsCTC494-Lm' and exp != 'V01' and exp != 'V05':
        if exp != 'LsCTC494-Lm' and exp != 'V05':
            # !! if diff model mu(pH) change 3*n_cl to 2*n_cl !!!
            param_ode_new[2*3 + 2 + 3+1] = 0.
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode_new, x_max[i])
        else:
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, x_max[i])
    ll_x = ll_x[ll_x != 0]
    return calibr_setup["aggregation_func"]([ll_x])


def sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0, param_ode, x_max):
    # TODO mb: do we need to fit also data for BAC, LA (pH)
    # Then obs_x -> obs_x+m
    # + pH instead of temp?
    model = calibr_setup["model"]
    days, [obs_x] = calibr_setup["data_array"]
    #temp = calibr_setup["exp_temps"][exp]
    pH_series = np.array([obs_x[i][-1], days]).T
    const = [pH_series, n_cl]

    C0 = set_initial_vals(x0, None, n_cl, pH0=obs_x[i][-1][0])
    #np.concatenate((np.array(x0), np.array([0., 0.]), np.array([1., 0., 0., 6.])))
    C = model_ODE_solution(model, days, param_ode, C0, const, t0=days[0])
    obs_model = observable(days, C)
    #ll_x0 = (obs_x[i][:-1] - C[:-1]) ** 2 / x_max[:-1]
    ll_x0 = [
        (obs_x[i][0] - obs_model[0]) ** 2 / x_max[0], #  G
        (obs_x[i][1] - obs_model[1]) ** 2 / x_max[1],
        (obs_x[i][2] - obs_model[2]) ** 2 / np.max(x_max[2]), # BAC
        (obs_x[i][3] - obs_model[3]) ** 2,# / np.max(x_max[3]), #/ x_max[3], # LA
        #(obs_x[i][4] - obs_model[4]) ** 2 / np.max(x_max[4]),  # pH
    ]
    return np.array(ll_x0)
