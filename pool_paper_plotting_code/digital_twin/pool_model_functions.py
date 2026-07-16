import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import json
from pydantic.json import custom_pydantic_encoder

from digital_twin.model import model_ODE_solution, optimization_func

########### In-silico data generation ############
def merge_dfs(dfs, sort=True):
    return pd.concat(dfs).groupby(level=0, sort=sort).sum()

def data_generation_poolpaper(n_cl, param_ode, x10, times, path=''):
    dfs_ode = []
    add_name = ''
    temps = [2.,]
    ntr = 1
    df_ode = model_wotemp(n_cl, temps, ntr, times, param_ode=param_ode, x10=x10, add_name=add_name,path=path, exp_start_offset=0)
    dfs_ode.append(df_ode)
    return df_ode

def model_wotemp(n_cl, temps, ntr, times, param_ode=None, x10=None, path='',add_name='', exp_start_offset=0):
    np.random.seed(46987)
    t = times
    if param_ode is None:
       print('No parameter vector provided.')
       exit()
    if x10 is not None:
        x0 = set_initial_vals(x10, None, n_cl)
    df_ode = generate_data_dfs(ode_model_coculture, t, np.array(param_ode), x0, temps, n_cl, n_traj=ntr, exp_start_offset=exp_start_offset)
    save_all_dfs([df_ode], names=[f'poolpaper{add_name}'], path=path)
    print(add_name, param_ode, '\n')
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode)}, f'Generated_param{add_name}.json', dir=path)
    return df_ode

def set_initial_vals(x10, temps, n_cl, pH0=6.):
    return np.concatenate((x10, [1., 0., 0., pH0]))

def generate_data_dfs(model, t, param, x0, temps, n_cl, n_traj=1, exp_start_offset=0):
    df_ode = []
    for j, temp in enumerate(temps):
        exp_start = exp_start_offset + 1 + j
        const = [[temp], n_cl]
        x0_exp = np.asarray(x0[j], dtype=float)
        param_ode = np.asarray(param[:n_cl*(4+n_cl)+2])
        x = model_ODE_solution(model, t, param_ode, x0_exp, const)#, jac=jac)
        bacteria_name = ['Ls', 'Lm']
        df_ode0 = merge_dfs([create_df_poolpaper(t, x, [f'V{j+exp_start:02d}'], bacteria_name, stds=0.) for j in range(n_traj)], sort=False)
        df_ode.append(df_ode0)
    return merge_dfs(df_ode, sort=False)

def save_all_dfs(dfs, names=[''], path=''):
    for df, n in zip(dfs, names):
        df.to_pickle(path+f'dataframe_{n}.pkl')


# Save the modell calibr_result to the json file
def json_dump(calibr_result, filename, dir='', **kwargs):
    # Special encoders for any object we might come across
    if "default" not in kwargs.keys():
        kwargs["default"] = _get_encoder(calibr_result)
    if "indent" not in kwargs.keys():
        kwargs["indent"] = 4

    # Return the json output as stri
    with open(dir+filename, "w") as f:
        json.dump(calibr_result, f, **kwargs)

def _get_encoder(calibr_result):
    encoders = {
        np.ndarray: lambda x: x.tolist(),
        np.int32: lambda x: str(x),
        np.int64: lambda x: "here",
    }
    # Define the encoder as a modification of the pydantic encoder
    return lambda obj: custom_pydantic_encoder(encoders, obj)

def read_from_json(filename, dir=''):
    f = open(dir + filename,)
    sol = json.load(f)
    return sol

def create_df_poolpaper(days, obs, name_part, bact_name, stds=0):
    n_cl = len(bact_name)
    n_states = 1
    data = {"Measurement": ['x_'+bact_name[i]+f'_State_{j:02d}' for i in range (n_cl) for j in range (n_states)]+['m_BAC', 'm_LA', 'pH']}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'poolpaper'])] = o
    df = pd.DataFrame(data=data).set_index('Measurement') 
    #df = pd.DataFrame(data=data).groupby('Measurement', sort=False).sum()
    return df

############### Read data from excel ######################3
def experimental_values(name, skiprows=0, path_data='', LA_sheetname='', path='', exp_start_offset=0):
    filename = 'CCD_results_counts_Part 2.xlsx'
    df_counts = pd.read_excel(path_data + filename, keep_default_na=True, sheet_name='R9_rep', skiprows=skiprows, usecols='A:F', nrows=16)
    # TODO: temporal solution to round all t to the round number: maybe not accurate: what else to do?
    time_count = df_counts['Time'].astype(int)
   
    Ls = np.array(df_counts['LAB (cfu/mL)'])
    Lm = np.array(df_counts['LM (cfu/mL)'])
    pH = np.array(df_counts['pH'])

    if name == 'Lm':
        Ls = np.array([0. for _ in range(len(Lm))])
    elif name == 'Ls23K' or name == 'LsCTC494':
        Lm = np.array([0. for _ in range(len(Ls))])
    elif name == 'LsCTC494-Lm':
        Lm[-1] = np.nan

    if name == 'Ls23K' or name == 'Lm' or name == 'Ls23K-Lm' or name == 'LsCTC494': # ???
        time_BAC = time_count
        BAC = np.array([0. for t in time_BAC])
    else:
        df_BAC = pd.read_excel(path_data+"/8_BA/BA_09.xlsx", keep_default_na=True, sheet_name='BA_prod', skiprows=19, usecols='M:Q', nrows=16)
        time_BAC = df_BAC['Time, h (1)'].astype(int)
        BAC = np.array(df_BAC['BA (10^3 AU/mL)'])*10**3

    df_LA = pd.read_excel(path_data + "/7_LA/RUN_09.xlsx", keep_default_na=True, sheet_name=LA_sheetname, skiprows=19, usecols='M:Q', nrows=16)
    time_LA = df_LA['Time, h (1)'].astype(int)
    # Restore missing measurements of LA
    time_all = sorted(set(list(time_count) + list(time_BAC) + list(time_LA)))
    df_LA["Time, h (1)"] = df_LA["Time, h (1)"].astype(int)
    df_LA_new = df_LA.T
    j = 0
    for t in time_all:
        if not np.any(np.abs(df_LA['Time, h (1)'] - t) <= 0.3):
            j = j+1
            df_LA_new[len(df_LA['Time, h (1)'])+j] = [t]+ [np.nan for _ in range (len(df_LA.columns)-1)]
    df_LA = df_LA_new.T.sort_values(by=['Time, h (1)'], ascending=True)
    LA = np.array(df_LA['LA (mg/mL)'])
    #LA[LA <= 0.0] = 0.0
    
    #Resource = [np.nan for _ in range (len(time_all))]
    obs = np.array([Ls, Lm, BAC, LA, pH])
    exp_start = exp_start_offset + 1
    df = create_df_poolpaper(time_all, obs, [f'V{exp_start:02d}'], ['Ls', 'Lm'])
    save_all_dfs([df], names=['poolpaper_' + name], path=path)
    return df


###############################################################################################

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

    ###############################################################3
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