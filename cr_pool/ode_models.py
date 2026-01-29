import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from dataclasses import dataclass

from .units import SECOND, MICRON, PICO_GRAM, MOL


def observable_2pool_2species(out):
    output = out.y
    return np.array(
        [
            output[0] + output[1],  # NA
            output[2] + output[3],  # NB
            output[0] + output[1] + output[2] + output[3],  # NA + NB
        ]
    )


@dataclass
class ODEParameters:
    # TODO adjust these parameters
    lag_phase_transition_rate_1: float = 0.0025
    lag_phase_transition_rate_2: float = 0.005
    production_rate_1: float = 0.002
    production_rate_2: float = 0.002
    inhibitor_production: float = 0.1
    inhibition_constant: float = 1e-7
    total_cell_volume: int = 2e5

    @property
    def parameters(self):
        return (
            self.lag_phase_transition_rate_1,  # lambd1,
            self.lag_phase_transition_rate_2,  # lambd2,
            self.production_rate_1,  # alpha1,
            self.production_rate_2,  # alpha2,
            self.inhibitor_production,  # kappa,
            self.inhibition_constant,  # muI,
            self.total_cell_volume,  # N_t
        )


@dataclass
class ODEInitialValues:
    initial_total_volume_lag_phase_1: float = 127.2 * MICRON**2
    initial_total_volume_growth_phase_1: float = 0
    initial_total_volume_lag_phase_2: float = 127.2 * MICRON**2
    initial_total_volume_growth_phase_2: float = 0
    # NEVER TOUCH THIS VALUE (IF YOU WANT TO GET TOTAL_CELL_VOLUME)!
    initial_nutrients: float = 1.0
    initial_inhibitor: float = 0

    @property
    def initial_values(self):
        return np.array(
            [
                self.initial_total_volume_lag_phase_1,
                self.initial_total_volume_growth_phase_1,
                self.initial_total_volume_lag_phase_2,
                self.initial_total_volume_growth_phase_2,
                self.initial_nutrients,
                self.initial_inhibitor,
            ]
        )


@dataclass
class ODEMetaParameters:
    time_start: float = 0 * SECOND
    time_dt: float = 0.5 * SECOND
    time_steps: int = 20_001
    time_save_interval: int = 500

    @property
    def time_eval(self):
        return np.arange(
            self.time_start,
            self.time_final + self.time_dt,
            self.time_save_interval * self.time_dt,
        )

    @property
    def time_final(self):
        return self.time_start + self.time_dt * self.time_steps


def pool_model_spatial_limit(t, x, params):
    (
        L1,
        G1,
        L2,
        G2,
        R,
        I,
    ) = x
    (
        lambd1,
        lambd2,
        alph1,
        alph2,
        kappa,
        muI,
        N_t,
    ) = params
    alpha2_inhib = alph2 / (1 + muI * I)
    return [
        -lambd1 * R * L1,
        lambd1 * R * L1 + alph1 * R * G1,
        -lambd2 * R * L2,
        lambd2 * R * L2 + alpha2_inhib * R * G2,
        -(alph1 / N_t) * R * G1 - (alpha2_inhib / N_t) * R * G2,
        kappa * G1,
    ]


@dataclass
class ODEModel(ODEInitialValues, ODEParameters, ODEMetaParameters):
    def solve_ode_raw(self):
        initial_values = self.initial_values
        params = self.parameters

        t0 = self.time_start
        t1 = self.time_final
        t_eval = self.time_eval

        sol_model = sp.integrate.solve_ivp(
            pool_model_spatial_limit,
            [t0, t1],
            initial_values,
            method="LSODA",
            t_eval=t_eval,
            args=(params,),
            rtol=2e-4,
        )
        return sol_model

    def solve_ode_observable(self, observable):
        ode_sol = self.solve_ode_raw()
        obs = observable(ode_sol)
        return obs


def model_solve(model, times, param, x0, const, obs_func=None):
    model_output = model_ODE_solution(model, times, param, x0, const)
    if obs_func is None:
        observables = model_output
    else:
        observables = np.array(obs_func(model_output))
    return times, observables


# The model solution for given ODEs
def model_ODE_solution(model, t, param, x0, const):
    sol_model = sp.integrate.solve_ivp(
        model, [0, t[-1]], x0, method="LSODA", t_eval=t, args=(param, x0, const), rtol=2e-4
    )
    return sol_model.y


if __name__ == "__main__":
    # 9. Spatial Limitation:
    Nt = 1e4
    x0sp = [5.0, 0.0, 5.0, 0.0, 1.0, 0.0]
    constsp = [
        0.008,
        0.005,
        4.0,
        5.0,
        0.1,
        0.1,
        Nt,
    ]  # (lambd1, lambd2, alph1, alph2, chi, muI, Nt, )
    times, observabl = model_solve(
        pool_model_spatial_limit,
        np.linspace(0, 10, 100),
        [],
        x0sp,
        constsp,
        obs_func=observable_2pool_2species,
    )

    fig, ax = plt.subplots()
    labels = ["Species 1", "Species 2", "Sp 1 + Sp 2"]
    for j, obs in enumerate(observabl):
        ax.plot(times, obs, label=labels[j])  # , color=colorsp[i+2])
    ax.set_xlabel("Time, [days]")
    ax.tick_params(labelsize=13)
    ax.set_ylabel("Total Bacterial Count")
    # ax.set_yscale('log')
    ax.set_xlim(0.0, 10.0)
    ax.legend()
    plt.savefig("pool_model_spatial.pdf", bbox_inches="tight")
    plt.close(fig)
