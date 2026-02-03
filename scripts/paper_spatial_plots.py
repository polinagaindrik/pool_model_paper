import cr_pool as crp
from pathlib import Path
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from cr_pool.units import (
    SECOND,
    MICRON,
    MOL,
    PICO_GRAM,
)

DIFFUSION_UNITS = "\\text{\\textmu m}^2\\text{s}^{-1}"

# Diffusion constants, randomness, homogenous
CONFIGS = [
    (20.0, 0, True, "Homogeneous"),
    (2.0, 0, True, "Homogeneous"),
    (20.0, 0, False, "Heterogeneous"),
    (20.0, 0.30, True, "Random"),
]


def calculate_results(diffusion_constant, randomness, homogenous):
    # Domain Settings
    domain = crp.Domain()
    domain.size = 1_000
    domain.diffusion_constants = [diffusion_constant] * 2
    domain.n_voxels = 50

    # Meta Parameters
    meta_params = crp.MetaParams()
    precision = 4
    meta_params.dt = SECOND / precision
    meta_params.n_times = 40_000 * precision + 1
    meta_params.save_interval = 1_000 * precision
    meta_params.n_threads = 8

    cell = crp.BacteriaTemplate()

    # Cellular Reactions
    cell.cellular_reactions.uptake_rate = 0.0025 / SECOND
    cell.cellular_reactions.inhibition_production_rate = 0.025 / SECOND
    cell.cellular_reactions.inhibition_coefficient = 0.1 * MICRON**3 / MOL
    cell.cellular_reactions.food_to_volume_conversion = 0.1 * MICRON**3 / MOL

    # Interaction
    cell.cellular_reactions.potential_strength = 0.125 * PICO_GRAM * MICRON / SECOND
    cell.cellular_reactions.cell_volume = np.pi * (1.5 * MICRON) ** 2
    cell.mechanics.damping_constant = 0.125 / SECOND
    cell.mechanics.mass = 1.09 * cell.cellular_reactions.cell_volume * PICO_GRAM / MICRON**2
    cell.cellular_reactions.potential_strength = 0.03125 * PICO_GRAM * MICRON / SECOND**2

    # Cell Cycle
    cell.cycle.lag_phase_transition_rate_1 = 0.001250 / SECOND
    cell.cycle.lag_phase_transition_rate_2 = 0.000625 / SECOND
    cell.cycle.volume_division_threshold = 2 * np.pi * (1.5 * MICRON) ** 2

    cells = crp.generate_cells(18, 18, domain, randomness, homogenous=homogenous, template=cell)

    output_path = crp.run_or_load_simulation(
        cells,
        domain,
        meta_params,
    )

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="crp_spatial_plots", description="Plots results needed for the paper publication"
    )
    parser.add_argument("paths", type=str, nargs="*")
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--n-threads", type=int, default=None)
    args = parser.parse_args()

    if args.n_threads is None:
        args.n_threads = mp.cpu_count()

    paths = args.paths
    paths = [(Path(p), None) for p in paths]

    if len(paths) == 0:
        paths = []
        for diffusion_constant, randomness, homogenous, title in CONFIGS:
            full_title = f"{title} $D={diffusion_constant}{DIFFUSION_UNITS}$"
            output_path = calculate_results(diffusion_constant, randomness, homogenous)
            paths.append((Path(output_path), full_title))

    dataframes = []
    pb = tqdm(total=len(paths) * 2, desc="Loading Data")
    for output_path, title in paths:
        domain, initial_cells, meta_params = crp.get_simulation_settings(output_path)
        data_cells, data_voxels = crp.analyze_all_cell_voxel_data(
            output_path,
            meta_params,
            args.n_threads,
            False,
            pb,
        )
        dataframes.append((data_cells, data_voxels))

    crp.load_style()
    pb = tqdm(total=len(paths) * 3, desc="Plotting Graphs")
    for (output_path, title), (data_cells, data_voxels) in zip(paths, dataframes):
        crp.plot_growth_curve(data_cells, output_path)
        pb.update(1)
        crp.plot_nutrients(data_voxels, output_path)
        pb.update(1)
        crp.plot_comparisons(data_cells, output_path, title=title)
        pb.update(1)

    if args.save_snapshots:
        pb = tqdm(
            total=len(paths) * 4,
            desc="Plotting Snapshots",
        )
        for output_path, _ in paths:
            iters = crp.get_all_iterations(output_path)
            crp.save_snapshot(output_path, iters[6], True, ["png", "pdf"])
            pb.update()
            crp.save_snapshot(output_path, iters[12], True, ["png", "pdf"])
            pb.update()
            crp.save_snapshot(output_path, iters[18], True, ["png", "pdf"])
            pb.update()
            crp.save_snapshot(output_path, iters[24], True, ["png", "pdf"])
            pb.update()
