#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp

# Define the pool model ODEs (general case)
def pool_model(t, x, param, x0, const):
    (alpha, lambd, n_max, ) = param
    (x01, ) = x0
    (x1, ) = x
    return [alpha * (x1 - x01 * np.exp(- lambd * t)) * (1 - x1 / n_max)]

# Define the pool model ODEs for 2 species model with resource competition between species
# With the waste production ofspecies 1, with waste-induced death of the species 2 
def pool_model_2species_resource_waste(t, x, param, x0, const):
    (L1, L2, G1, G2, R, W1,) = x
    (alph1, alph2, lambd1, lambd2, chi1, muW2, Nt, ) = const
    
    return [
        - lambd1 * R * L1,
        - lambd2 * R * L2,
          lambd1 * R * L1 + alph1 * R * G1,
          lambd2 * R * L2 + alph2 * R * G2 - muW2/(chi1*Nt) * W1 * G2,
        - alph1/Nt * R * G1 - alph2/Nt * R * G2,
          chi1 * G1
    ]


# The model solution for given ODEs
def model_ODE_solution(model, t, param, x0, const):
    sol_model = solve_ivp(model, [0, t[-1]], x0, method='LSODA', t_eval=t, args=(param, x0, const), rtol=2e-4)
    return sol_model.y