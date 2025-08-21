#!/usr/bin/env python3
import numpy as np
import itertools as iter

from digital_twin.model import model_ODE_solution


def generate_insilico_data(model, times, param_arr, prob_arr, x0_arr, x0_prob_arr=None, const=[], obs_func=None,
                           n_traj=5,  noise=.0, rel_noise=.0, add_name=''):
    # Convert probabilities and param array to flatten format
    param_arr_flatten, prob_arr_flatten = [], []
    if len(param_arr) != 0:
        for ind in iter.product(*[[i for i in range(np.shape(param_arr)[-1])] for _ in range(np.shape(param_arr)[0])]):  
            param_arr_flatten.append([param_arr[j][ind[j]] for j in range (len(ind))])
            prob_arr_flatten.append(prob_arr[*ind])
    prob_arr_flatten = np.array(prob_arr_flatten)
    data = []
    for j in range (n_traj):
        param_new = sample_parameter_from_distribution(param_arr_flatten, prob_arr_flatten, size=1)
        if x0_prob_arr is None:
            x0_new = [x0[0] for x0 in x0_arr]
        else:
            x0_new = [sample_parameter_from_distribution(x0, x0_prob, size=1) for x0, x0_prob in zip(x0_arr, x0_prob_arr)]
        # Generate timepoints if not given:
        if type(times) == tuple:
            times_generated = generate_discrete_timepoints(*times)
        else:
            times_generated = times[j]
        model_output = model_ODE_solution(model, times_generated, param_new, x0_new, const)
        if obs_func is None:
            observables = model_output + np.random.normal(0.0, rel_noise*model_output+noise, size=np.shape(model_output))
        else:
            observables = np.array(obs_func(model_output))
            observables = observables + np.random.normal(0.0, rel_noise*observables+noise, size=np.shape(observables))
        data.append({
            'experiment_name': 'In-silico' + add_name,
            'times': np.array(times_generated),
            'obs_mean': observables,
            'obs_std': np.abs(rel_noise * observables + noise),
            'obs_whole': [[[tr] for tr in obs] for obs in observables],
            'model_output': model_output,
            'const' : const,
            'other': [param_new, x0_new] # Save parameters and init conditions for generated curve
            })
    return data


# The list of possible combinations of parameter combinations, 1D array of probabilities
def sample_parameter_from_distribution(param_flatten, prob_flatten, size=1):
    if len(param_flatten) != 0:
        index_sampled = np.random.choice(np.linspace(0, len(prob_flatten)-1, len(prob_flatten), dtype=int), size=size, 
                                         p=prob_flatten/np.sum(prob_flatten))[0]
        return param_flatten[index_sampled]   
    else:
        return param_flatten


# Generate measurement time points fro each trajectory (curve)
def generate_discrete_timepoints(time_bnds, n_times):
    time = np.random.choice(np.linspace(time_bnds[0], time_bnds[1], int(time_bnds[1]-time_bnds[0]+1)), 
                            size=(n_times), replace=False)
    return np.sort(time, axis=0)