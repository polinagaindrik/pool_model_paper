# fmt: off
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/pool_paper_plotting_code")
from digital_twin import *


def model_solve(model, times, param, x0, const, obs_func=None):
    model_output = model_ODE_solution(model, times, param, x0, const)
    if obs_func is None:
        observables = model_output
    else:
        observables = np.array(obs_func(model_output))
    return times, observables


# The model solution for given ODEs
def model_ODE_solution(model, t, param, x0, const):
    sol_model = solve_ivp(model, [0, t[-1]], x0, method='LSODA', t_eval=t, args=(param, x0, const), rtol=2e-4)
    return sol_model.y

def pool_model_spatial_limit(t, x, temp, x0, const):
    (L1, G1, L2, G2, R, I,) = x
    (lambd1, lambd2, alph1, alph2, chi, muI, N_t, ) = const
    res = 1 - (L1+L2+G1+G2)/N_t
    alpha2_inhib = alph2/(1 + muI*I)
    return [
        - lambd1 * R * L1,
          lambd1 * R * L1 + alph1 * R * G1,
        - lambd2 * R * L2,
          lambd2 * R * L2 + alpha2_inhib * R * G2,
        - (alph1/N_t) * R * G1 - (alpha2_inhib/  N_t) * R * G2,
          chi * G1
    ]

def observable_2pool_2species(output):
    return [output[0]+output[1], # NA
            output[2]+output[3], # NB
            output[0]+output[1]+output[2]+output[3]] # NA + NB

if __name__ == "__main__":
    # 9. Spatial Limitation:
    Nt = 1e4
    x0sp = [5., 0., 5., 0., 1., 0.]
    constsp = [.008, .005, 4., 5., .1, .1, Nt] # (lambd1, lambd2, alph1, alph2, chi, muI, Nt, )
    times, observabl = model_solve(pool_model_spatial_limit, np.linspace(0, 10, 100), [], x0sp, constsp,
                             obs_func=observable_2pool_2species) 

    fig, ax = plt.subplots()
    labels = ['Species 1', 'Species 2', 'Sp 1 + Sp 2']
    for j, obs in enumerate(observabl):
        ax.plot(times, obs, label=labels[j])#, color=colorsp[i+2])
    ax.set_xlabel('Time, [days]')
    ax.tick_params(labelsize=13)
    ax.set_ylabel('Total Bacterial Count')
    #ax.set_yscale('log')
    ax.set_xlim(0., 10.)
    ax.legend()
    plt.savefig('pool_model_spatial.pdf', bbox_inches='tight')
    plt.close(fig)
