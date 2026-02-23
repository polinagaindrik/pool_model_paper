from argparse import Namespace
from types import SimpleNamespace
import numpy as np
import scipy as sp
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

import cr_pool as crp
from scripts.paper_spatial_plots import calculate_results, CONFIGS, DIFFUSION_UNITS


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


def print_ns(ns, ident=1):
    for k in ns.__dict__:
        elem = ns.__dict__[k]
        if type(elem) is not SimpleNamespace:
            print(f"{'    ' * ident}{k}: {elem}")
        else:
            print(f"{'    ' * ident}{k}", "{")
            print_ns(elem, ident + 1)
            print("    " * ident + "}")


def solve_ode(opath, full_title):
    domain, initial_cells, meta_params = crp.get_simulation_settings(opath)
    data_cells, data_voxels = crp.analyze_all_cell_voxel_data(
        opath,
        meta_params,
        30,
        False,
    )
    cells_A = [c for c in initial_cells if c.cellular_reactions.species == "S1"]
    cells_B = [c for c in initial_cells if c.cellular_reactions.species == "S2"]
    cell_A = cells_A[0]
    cell_B = cells_B[0]

    domain_volume = domain.size**2
    sigma = cell_A.cellular_reactions.food_to_volume_conversion
    N_T = domain.initial_concentrations[0]
    lambda_A = cell_A.cycle.lag_phase_transition_rate_1
    lambda_B = cell_B.cycle.lag_phase_transition_rate_2
    mu_A = cell_A.cellular_reactions.uptake_rate * sigma * N_T
    mu_B = cell_B.cellular_reactions.uptake_rate * sigma * N_T
    kappa = sigma * cell_B.cellular_reactions.inhibition_production_rate
    nu = cell_B.cellular_reactions.inhibition_coefficient

    parameters = (
        lambda_A,
        lambda_B,
        mu_A,
        mu_B,
        kappa,
        nu,
        N_T,
    )

    # Construct ODE
    t0 = meta_params.t_start
    t1 = t0 + meta_params.n_times * meta_params.dt
    dt = meta_params.save_interval * meta_params.dt
    t_eval = np.arange(t0, t1, dt)

    vol_average = np.mean(data_cells["bacteria_volume_1"] / data_cells["bacteria_count_1"])
    init_l1 = len(cells_A) / domain_volume * vol_average / sigma
    init_l2 = len(cells_B) / domain_volume * vol_average / sigma

    initial_values = np.array([init_l1, 0, init_l2, 0, 1, 0])

    res = sp.integrate.solve_ivp(
        pool_model_spatial_limit,
        [t0, t1],
        initial_values,
        method="LSODA",
        t_eval=t_eval,
        args=(parameters,),
        rtol=2e-4,
    )

    y1 = res.y[1] * domain_volume / vol_average * sigma
    y2 = res.y[3] * domain_volume / vol_average * sigma
    y3 = y1 + y2

    fig, ax = plt.subplots()
    ax.plot(res.t / crp.HOUR, y1, color=crp.CA, label="Species A", alpha=0.5)
    ax.plot(res.t / crp.HOUR, y2, color=crp.CB, label="Species B", alpha=0.5)
    ax.plot(res.t / crp.HOUR, y3, color=crp.CN, label="Combined", alpha=0.5)

    y1 = data_cells["bacteria_volume_1"] / vol_average
    y2 = data_cells["bacteria_volume_2"] / vol_average
    y3 = y1 + y2

    ax.plot(data_cells["time"], y1, color=crp.CA, linestyle=":")
    ax.plot(data_cells["time"], y2, color=crp.CB, linestyle=":")
    ax.plot(data_cells["time"], y3, color=crp.CN, linestyle=":")

    y1 = data_cells["bacteria_count_1"]
    y2 = data_cells["bacteria_count_2"]
    y3 = y1 + y2

    ax.plot(data_cells["time"], y1, color=crp.CA, linestyle="--", alpha=0.6)
    ax.plot(data_cells["time"], y2, color=crp.CB, linestyle="--", alpha=0.6)
    ax.plot(data_cells["time"], y3, color=crp.CN, linestyle="--", alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[:3] + [
        mpl.lines.Line2D([0], [0], color="gray", linestyle="-", lw=2),
        mpl.lines.Line2D([0], [0], color="gray", linestyle="--", lw=2),
        mpl.lines.Line2D([0], [0], color="gray", linestyle=":", lw=2),
    ]
    labels = labels[:3] + ["ODE", "ABM (Count)", "ABM (Vol)"]
    ax.legend(handles, labels)
    ax.set_title(full_title)

    ax.set_ylabel("Number of Cells")
    ax.set_xlabel("Time [h]")

    fig.savefig(opath / "abm_ode_comparison-cellcount.png")
    fig.savefig(opath / "abm_ode_comparison-cellcount.pdf")


if __name__ == "__main__":
    crp.load_style()

    paths = []
    for diffusion_constant, randomness, homogenous, title in CONFIGS:
        full_title = f"{title} $D={diffusion_constant}{DIFFUSION_UNITS}$"
        output_path = calculate_results(diffusion_constant, randomness, homogenous)
        paths.append((Path(output_path), full_title))

    for opath, full_title in paths:
        solve_ode(opath, full_title)
