#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp

# Define the pool model ODEs (general case)
def pool_model(t, x, param, x0, const):
    (alpha, lambd, n_max, ) = param
    (x01, ) = x0
    (x1, ) = x
    return [alpha * (x1 - x01 * np.exp(- lambd * t)) * (1 - x1 / n_max)]


# The model solution for given ODEs
def model_ODE_solution(model, t, param, x0, const):
    sol_model = solve_ivp(model, [0, t[-1]], x0, method='LSODA', t_eval=t, args=(param, x0, const), rtol=2e-4)
    return sol_model.y