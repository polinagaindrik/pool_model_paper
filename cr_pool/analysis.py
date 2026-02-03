import numpy as np
import multiprocessing as mp
import pandas as pd
import tqdm

from .simulation import (
    get_all_iterations,
    get_elements_at_iter,
    calculate_entropy,
)

from .units import HOUR


def print_cell_properties(output_path):
    # Cellular Properties and their names in dataframe
    iter_0_particles = get_elements_at_iter(output_path, 0)
    print("Name".ljust(62), "Type")
    print("⎯" * 80)
    for ty, col in zip(
        iter_0_particles.dtypes,
        iter_0_particles.columns,
    ):
        print(
            "{}".format(col).ljust(62),
            "{}".format(ty),
        )


def print_domain_properties(output_path):
    # Voxel Properties and their names in dataframe
    iter_0_voxels = get_elements_at_iter(
        output_path,
        0,
        element_path="voxel_storage",
    )
    print("Name".ljust(62), "Type")
    print("⎯" * 80)
    for ty, col in zip(
        iter_0_voxels.dtypes,
        iter_0_voxels.columns,
    ):
        print(
            "{}".format(col).ljust(62),
            "{}".format(ty),
        )


def analyze_voxel_data(output_path, iteration, meta_params, get_entropy):
    # Calculate total extracellular concentrations
    entry = get_elements_at_iter(
        output_path,
        iteration,
        element_path="voxel_storage",
    )
    mi = np.array([x for x in entry["element.voxel.min"]])
    ma = np.array([x for x in entry["element.voxel.max"]])
    diff = np.abs(ma - mi)
    conc = np.array([c for c in entry["element.voxel.extracellular_concentrations"]])
    volume = np.prod(diff, axis=1)
    conc_total = np.sum(volume[:, None] * conc, axis=0)

    return {
        "iteration": iteration,
        "time": iteration * meta_params.dt / HOUR,
        "nutrients_conc_total": conc_total[0],
        "inhib_total": conc_total[1],
    }


def _analyze_voxel_data_helper(args):
    return analyze_voxel_data(*args)


def analyze_cell_data(output_path, iteration, meta_params, get_entropy):
    entry = get_elements_at_iter(output_path, iteration)

    # Calculate number of cells for species
    bacteria_1 = entry[entry["element.cell.cellular_reactions.species"] == "S1"]
    bacteria_2 = entry[entry["element.cell.cellular_reactions.species"] == "S2"]
    bacteria_count_1 = len(bacteria_1)
    bacteria_count_2 = len(bacteria_2)

    # Calculate the number of bacteria in lag phase
    bacteria_in_lag_phase_1 = len(
        bacteria_1[bacteria_1["element.cell.cellular_reactions.lag_phase_active"] == True]
    )
    bacteria_in_lag_phase_2 = len(
        bacteria_2[bacteria_2["element.cell.cellular_reactions.lag_phase_active"] == True]
    )

    # Calculate total intracellular concentrations
    volume = np.array([x for x in entry["element.cell.cellular_reactions.cell_volume"]])

    volume_1 = np.sum(volume[entry["element.cell.cellular_reactions.species"] == "S1"])
    volume_2 = np.sum(volume[entry["element.cell.cellular_reactions.species"] == "S2"])
    total_cell_volume = np.sum(volume)

    # Return combined results
    res = {
        "iteration": iteration,
        "time": iteration * meta_params.dt / HOUR,
        "bacteria_count_1": bacteria_count_1,
        "bacteria_count_2": bacteria_count_2,
        "bacteria_count_total": bacteria_count_1 + bacteria_count_2,
        "bacteria_in_lag_phase_1": bacteria_in_lag_phase_1,
        "bacteria_in_lag_phase_2": bacteria_in_lag_phase_2,
        "bacteria_in_lag_phase_total": bacteria_in_lag_phase_1 + bacteria_in_lag_phase_2,
        "bacteria_volume_1": volume_1,
        "bacteria_volume_2": volume_2,
        "bacteria_volume_total": total_cell_volume,
    }

    # Calculate the entropy at this step
    if get_entropy:
        res["entropy"] = calculate_entropy(output_path, iteration)

    return res


def _analyze_cell_data_helper(args):
    return analyze_cell_data(*args)


def analyze_all_cell_voxel_data(
    output_path,
    meta_params,
    n_threads=None,
    get_entropy=False,
    pb=None,
):
    # Construct pool to process results in parallel
    if n_threads is None:
        n_threads = mp.cpu_count()
    pool = mp.Pool(n_threads)

    # Gather args and use pool to get already pre-processed results
    args = [
        (output_path, iteration, meta_params, get_entropy)
        for iteration in get_all_iterations(output_path)
    ]
    data_cells = (
        pd.DataFrame(
            pool.imap_unordered(
                _analyze_cell_data_helper,
                args,
            ),
        )
        .sort_values("iteration")
        .reset_index(drop=True)
    )
    if pb is not None:
        pb.update()
    data_voxels = (
        pd.DataFrame(
            pool.imap_unordered(
                _analyze_voxel_data_helper,
                args,
            ),
        )
        .sort_values("iteration")
        .reset_index(drop=True)
    )
    if pb is not None:
        pb.update()

    # Close the pool to free any remaining resources
    pool.close()
    pool.join()

    return data_cells, data_voxels
