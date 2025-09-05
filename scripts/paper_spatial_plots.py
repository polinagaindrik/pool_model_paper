import cr_pool as crp
from pathlib import Path
import argparse

# Diffusion constants, randomness, homogenous
CONFIGS = [
    (30, 0, True),
    (5, 0, True),
    (30, 0, False),
    (30, 0.30, True),
]


def calcualte_results(diffusion_constant, randomness, homogenous):
    # Domain Settings
    domain = crp.Domain()
    domain.size = 1_000
    # domain.diffusion_constants = [5.0, 5.0]
    domain.diffusion_constants = [diffusion_constant] * 2

    # Meta Parameters
    meta_params = crp.MetaParams()
    meta_params.save_interval = 1_000
    meta_params.n_times = 40_001
    meta_params.dt = 0.25
    meta_params.n_threads = 8

    cells = crp.generate_cells(18, 18, domain, randomness, homogenous=homogenous)

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
    args = parser.parse_args()

    paths = args.paths

    if len(paths) == 0:
        paths = []
        for diffusion_constant, randomness, homogenous in CONFIGS:
            output_path = calcualte_results(diffusion_constant, randomness, homogenous)
            paths.append(output_path)

    paths = [Path(p) for p in paths]

    if args.save_snapshots:
        for output_path in paths:
            picked_iter = crp.get_all_iterations(output_path)[20]
            crp.save_snapshot(output_path, picked_iter)
        # for output_path in paths:
        #     crp.save_all_snapshots(Path(output_path))

    crp.load_style()
    for output_path in paths:
        domain, initial_cells, meta_params = crp.get_simulation_settings(output_path)
        data_cells, data_voxels = crp.analyze_all_cell_voxel_data(output_path, meta_params)
        crp.plot_growth_curve(data_cells, output_path)
        crp.plot_nutrients(data_voxels, output_path)
        crp.plot_comparisons(data_cells, output_path)
