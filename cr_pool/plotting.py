import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tqdm
import multiprocessing as mp
import gc
import os
from pathlib import Path

from .analysis import UNIT_DAY, UNIT_HOUR
from .ode_models import ODEModel, observable_2pool_2species
from .plotting_templates import COLORS_ALL, FIGSIZE_DEFAULT
from .cr_pool import Domain, BacteriaTemplate, MetaParams
from .simulation import (
    get_simulation_settings,
    get_all_iterations,
    get_elements_at_iter,
    calculate_spatial_density,
)

CA = COLORS_ALL["N_A"]
CB = COLORS_ALL["N_B"]
CN = COLORS_ALL["N"]


def plot_growth_curve(data_cells, output_path):
    # Growth Curve
    fig, ax1 = plt.subplots(figsize=FIGSIZE_DEFAULT)

    data_cells.plot(x="time", y="bacteria_count_1", ax=ax1, label="Species 1", color=CA, alpha=0.5)
    data_cells.plot(x="time", y="bacteria_count_2", ax=ax1, label="Species 2", color=CB, alpha=0.5)
    data_cells.plot(
        x="time", y="bacteria_count_total", ax=ax1, label="Combined", color=CN, alpha=0.5
    )
    ax1.set_ylabel("Bacteria Count")
    ax1.set_xlabel("Time [days]")

    ax2 = ax1.twinx()
    data_cells.plot(x="time", y="bacteria_volume_1", ax=ax2, label="Species 1", color=CA)
    data_cells.plot(x="time", y="bacteria_volume_2", ax=ax2, label="Species 2", color=CB)
    data_cells.plot(x="time", y="bacteria_volume_total", ax=ax2, label="Combined", color=CN)
    ax2.set_ylabel("Combined Bacteria Volume")

    ax1.legend()
    fig.tight_layout()
    fig.savefig(f"{output_path}/cell_growth.png")
    fig.savefig(f"{output_path}/cell_growth.pdf")
    plt.show(fig)


def plot_nutrients(data_voxels, output_path):
    # Nutrients
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    data_voxels.plot(
        x="time", y="nutrients_conc_total", ax=ax, label="Extracellular Nutrients", color=CN
    )
    ax.set_ylabel("Total Nutrients")
    ax.set_xlabel("Time [days]")
    fig.savefig(f"{output_path}/nutrients.png")
    fig.savefig(f"{output_path}/nutrients.pdf")


def abm_to_ode(
    domain: Domain, initial_cells: list[BacteriaTemplate], meta_params: MetaParams
) -> ODEModel:
    kwargs = {}

    # Define Initial Values
    mask = np.array([c.cellular_reactions.species == "S1" for c in initial_cells])
    volumes = np.array([c.cellular_reactions.cell_volume for c in initial_cells])
    bacteria_volume_1 = np.sum(volumes[mask])
    bacteria_volume_2 = np.sum(volumes[mask == False])

    kwargs["initial_total_volume_lag_phase_1"] = bacteria_volume_1
    kwargs["initial_total_volume_lag_phase_2"] = bacteria_volume_2

    # Define OdeParameters

    # lag phase
    kwargs["lag_phase_transition_rate_1"] = np.average(
        [c.cycle.lag_phase_transition_rate_1 for c in initial_cells]
    )
    kwargs["lag_phase_transition_rate_2"] = np.average(
        [c.cycle.lag_phase_transition_rate_2 for c in initial_cells]
    )

    # Inhibitor Production rate
    abm_cell_volume_max = np.average([c.cycle.volume_division_threshold for c in initial_cells])
    abm_cell_volume_min = abm_cell_volume_max / 2.0
    abm_cell_volume_average = 0.5 * (abm_cell_volume_max + abm_cell_volume_min)
    abm_domain_volume = domain.size**2
    abm_inhibition_production_rate = np.average(
        [c.cellular_reactions.inhibition_production_rate for c in initial_cells]
    )
    kwargs["inhibitor_production"] = abm_inhibition_production_rate

    abm_inhibition_coefficient = np.average(
        [c.cellular_reactions.inhibition_coefficient for c in initial_cells]
    )

    kwargs["inhibition_constant"] = abm_inhibition_coefficient / abm_domain_volume
    abm_initial_food = abm_domain_volume * domain.initial_concentrations[0]
    abm_food_to_volume = np.average(
        [c.cellular_reactions.food_to_volume_conversion for c in initial_cells]
    )

    # total cell volume instead of number of cells
    kwargs["total_cell_volume"] = abm_initial_food * abm_food_to_volume

    uptake_rates = np.array([c.cellular_reactions.uptake_rate for c in initial_cells])
    abm_uptake_rate_1 = np.average(uptake_rates[mask])
    abm_uptake_rate_2 = np.average(uptake_rates[mask == False])

    kwargs["production_rate_1"] = (
        abm_uptake_rate_1 * abm_food_to_volume * domain.initial_concentrations[0]
    )
    kwargs["production_rate_2"] = (
        abm_uptake_rate_2 * abm_food_to_volume * domain.initial_concentrations[0]
    )

    # Meta Parameters
    kwargs["time_start"] = meta_params.t_start
    kwargs["time_dt"] = meta_params.dt
    kwargs["time_steps"] = meta_params.n_times
    kwargs["time_save_interval"] = meta_params.save_interval

    # Define OdeMetaParameters
    ode_model = ODEModel(**kwargs)
    return ode_model


def calculate_difference(data_cells, output_path):
    settings = get_simulation_settings(output_path)

    ode_model = abm_to_ode(*settings)
    res = ode_model.solve_ode_raw()
    obs_ode = observable_2pool_2species(res)

    obs_abm = (
        np.array([data_cells[["bacteria_volume_1", "bacteria_volume_2", "bacteria_volume_total"]]])
        .reshape((obs_ode.T.shape))
        .T
    )

    variance = np.average(((obs_ode - obs_abm) / ode_model.total_cell_volume) ** 2) ** 0.5
    return variance


def plot_comparisons(data_cells, output_path):
    settings = get_simulation_settings(output_path)

    model = abm_to_ode(*settings)
    print(model)
    res = model.solve_ode_raw()
    obs = observable_2pool_2species(res)

    fig, ax = plt.subplots()

    ax.plot(res.t / UNIT_DAY, obs[0], color=CA, alpha=0.5, label="Species A (ODE)")
    ax.plot(res.t / UNIT_DAY, obs[1], color=CB, alpha=0.5, label="Species B (ODE)")
    ax.plot(res.t / UNIT_DAY, obs[2], color=CN, alpha=0.5, label="Combined (ODE)")

    data_cells.plot(
        x="time", y="bacteria_volume_1", ax=ax, label="Species A (ABM)", color=CA, linestyle="--"
    )
    data_cells.plot(
        x="time", y="bacteria_volume_2", ax=ax, label="Species B (ABM)", color=CB, linestyle="--"
    )
    data_cells.plot(
        x="time", y="bacteria_volume_total", ax=ax, label="Combined (ABM)", color=CN, linestyle="--"
    )

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Combined Bacteria Volume")
    # ax.set_yscale('log')
    ax.legend()
    fig.savefig(output_path / "abm_ode_comparison.png")
    fig.savefig(output_path / "abm_ode_comparison.pdf")
    fig.savefig(output_path / "abm_ode_comparison.eps")
    plt.show(fig)


def plot_lag_phase(data_cells, res, output_path):
    # Lag Phase
    fig, ax = plt.subplots()
    ax.set_title("Cells in Lag-Phase")

    ode_lag_phase_1 = res.y[0]
    ode_lag_phase_2 = res.y[2]
    ode_lag_phase_combined = res.y[0] + res.y[2]

    ax.plot(
        res.t / UNIT_DAY,
        ode_lag_phase_1 / (np.pi * 1.5**2),
        label="Species 1 (ODE)",
        color=CA,
        linestyle="--",
    )
    ax.plot(
        res.t / UNIT_DAY,
        ode_lag_phase_2 / (np.pi * 1.5**2),
        label="Species 2 (ODE)",
        color=CB,
        linestyle="--",
    )
    ax.plot(
        res.t / UNIT_DAY,
        ode_lag_phase_combined / (np.pi * 1.5**2),
        label="Combined (ODE)",
        color=CN,
        linestyle="--",
    )

    data_cells.plot(
        x="time", y="bacteria_in_lag_phase_1", ax=ax, label="Species 1 (ABM)", color="#252B33"
    )
    data_cells.plot(
        x="time", y="bacteria_in_lag_phase_2", ax=ax, label="Species 2 (ABM)", color="#D14027"
    )
    data_cells.plot(
        x="time", y="bacteria_in_lag_phase_total", ax=ax, label="Combined (ABM)", color="#7E57A5"
    )

    ax.legend()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Bacteria Volume")

    fig.tight_layout()
    fig.savefig(f"{output_path}/lag_phase.png")
    fig.savefig(f"{output_path}/lag_phase.pdf")
    plt.show(fig)


def plot_entropy(data_cells, output_path):
    # Entropy
    entropies = np.array([x for x in data_cells["entropy"]])

    fig, ax = plt.subplots()
    ax.set_title("Spatial Density Entropy")
    ax.plot(data_cells["time"], entropies[:, 0], label="Species 1", color="#D06062")
    ax.plot(data_cells["time"], entropies[:, 1], label="Species 2", color="#4E89B1")
    ax.legend()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Shannon Entropy")
    fig.tight_layout()
    fig.savefig(output_path / "entropy.png")
    fig.savefig(output_path / "entropy.pdf")
    plt.show(fig)


def _determine_image_save_path(output_path, iteration, fmt="png"):
    # Save images in dedicated images folder
    save_folder = Path(output_path) / "images"
    # Create folder if it does not exist
    save_folder.mkdir(parents=True, exist_ok=True)

    # Create save path from new save folder
    save_path = save_folder / "snapshot_{:08}.{}".format(iteration, fmt)

    return save_path


def _create_base_canvas(domain):
    # Define limits for domain from simulation settings
    xlims = np.array([0.0, domain.size])
    ylims = np.array([0.0, domain.size])

    # Define the overall size of the figure and adapt to if the domain was not symmetric
    figsize_x = 16
    figsize_y = (ylims[1] - ylims[0]) / (xlims[1] - xlims[0]) * figsize_x

    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Sets correct boundaries for our domain
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

    # Hide axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def _create_nutrient_voxel_color_mapping(domain):
    # Plot rectangles for background
    # Create color mapper for background
    nutrients_min = -0.2 * domain.initial_concentrations[0]
    nutrients_max = domain.initial_concentrations[0]
    norm2 = mpl.colors.Normalize(
        vmin=nutrients_min,
        vmax=nutrients_max,
        clip=True,
    )
    return mpl.cm.ScalarMappable(norm=norm2, cmap=mpl.cm.grey)


# Helper function to plot a single rectangle patch onto the canvas
def _plot_rectangle(entry, mapper, ax):
    x_min = entry["element.voxel.min"]
    x_max = entry["element.voxel.max"]
    conc = entry["element.voxel.extracellular_concentrations"][0]

    xy = [x_min[0], x_min[1]]
    dx = x_max[0] - x_min[0]
    dy = x_max[1] - x_min[1]
    color = mapper.to_rgba(conc) if not np.isnan(conc) else "red"
    rectangle = mpl.patches.Rectangle(xy, width=dx, height=dy, color=color)
    ax.add_patch(rectangle)


def _plot_labels(fig, ax, n_bacteria_1, n_bacteria_2):
    # Calculate labels and dimensions for length scale
    x_locs = ax.get_xticks()
    y_locs = ax.get_yticks()

    x_low = np.min(x_locs[x_locs > 0]) / 4
    y_low = np.min(y_locs[y_locs > 0]) / 4
    dy = (y_locs[1] - y_locs[0]) / 8
    dx = (x_locs[1] - x_locs[0]) / 1.5

    # Plot three rectangles as a length scale
    rectangle1 = mpl.patches.Rectangle(
        [x_low, y_low],
        width=dx,
        height=dy,
        facecolor="grey",
        edgecolor="black",
    )
    rectangle2 = mpl.patches.Rectangle(
        [x_low + dx, y_low],
        width=dx,
        height=dy,
        facecolor="white",
        edgecolor="black",
    )
    rectangle3 = mpl.patches.Rectangle(
        [x_low + 2 * dx, y_low],
        width=dx,
        height=dy,
        facecolor="grey",
        edgecolor="black",
    )

    # Plot two rectangles to display species count
    rectangle4 = mpl.patches.Rectangle(
        [x_low, dy + y_low],
        width=1.5 * dx,
        height=dy,
        facecolor="white",
        edgecolor="black",
    )
    rectangle5 = mpl.patches.Rectangle(
        [x_low + 1.5 * dx, dy + y_low],
        width=1.5 * dx,
        height=dy,
        facecolor="white",
        edgecolor="black",
    )

    # Add all rectangles to ax
    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)
    ax.add_patch(rectangle3)
    ax.add_patch(rectangle4)
    ax.add_patch(rectangle5)

    # Display size of the
    ax.annotate(
        f"{dx:5.0f}µm",
        (x_low + dx / 2, y_low + dy / 2),
        ha="center",
        va="center",
    )
    ax.annotate(
        f"{dx:5.0f}µm",
        (x_low + 3 * dx / 2, y_low + dy / 2),
        ha="center",
        va="center",
    )
    ax.annotate(
        f"{dx:5.0f}µm",
        (x_low + 5 * dx / 2, y_low + dy / 2),
        ha="center",
        va="center",
    )

    # Create rectangle to show number of agents
    ax.annotate(
        f"Species 1 {n_bacteria_1:5.0f}",
        (
            x_low + 1.5 * dx / 2,
            y_low + 3 * dy / 2,
        ),
        ha="center",
        va="center",
    )
    ax.annotate(
        f"Species 2 {n_bacteria_2:5.0f}",
        (
            x_low + 4.5 * dx / 2,
            y_low + 3 * dy / 2,
        ),
        ha="center",
        va="center",
    )


def _plot_voxels(df_voxels, ax, mapper):
    # Applies the previously defined plot_rectangle function to all voxels.
    df_voxels.apply(
        lambda entry: _plot_rectangle(entry, mapper, ax),
        axis=1,
    )


def _save_psychic_image(output_path, iteration):
    data = get_elements_at_iter(output_path, iteration)
    domain, _, _ = get_simulation_settings(output_path)
    fig, ax = _create_base_canvas(domain)

    data1 = data[data["element.cell.cellular_reactions.species"] == "S1"]
    data2 = data[data["element.cell.cellular_reactions.species"] != "S1"]

    Z1 = calculate_spatial_density(data1, domain, weights=True)
    Z2 = calculate_spatial_density(data2, domain, weights=True)

    Z = Z1 / (Z1 + Z2)

    ax.imshow(np.rot90(Z), cmap=plt.cm.viridis, extent=[0.0, domain.size, 0.0, domain.size])

    df_cells = get_elements_at_iter(output_path, iteration)

    _plot_bacteria(df_cells, ax)

    ax.set_xlim([0.0, domain.size])
    ax.set_ylim([0.0, domain.size])

    # ax.set_title(f"Entropy {sp.stats.entropy(Z.reshape(-1)):4.2f}")
    _plot_labels(fig, ax, n_bacteria_1=len(data1), n_bacteria_2=len(data2))

    save_path = _determine_image_save_path(output_path, iteration)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)


def _save_psychic_image_helper(args):
    _save_psychic_image(*args)


def save_all_psychich_images(output_path):
    args = [(output_path, iteration) for iteration in get_all_iterations(output_path)]
    _ = list(tqdm.tqdm(mp.Pool().imap_unordered(_save_psychic_image_helper, args), total=len(args)))


def _plot_bacteria(df_cells, ax):
    # Get positions as large numpy array
    positions = np.array([np.array(x) for x in df_cells["element.cell.mechanics.pos"]])
    s = (
        np.array([x for x in df_cells["element.cell.cellular_reactions.cell_volume"]]) / np.pi
    ) ** 0.5
    c = [
        "#24398c" if x else "#8c2424"
        for x in df_cells["element.cell.cellular_reactions.species"] == "S1"
    ]

    # Plot circles for bacteria
    for pos, si, ci in zip(positions, s, c):
        if si is not None:
            circle = plt.Circle(
                pos,
                radius=si,
                facecolor=ci,
                edgecolor=ci,
            )
            ax.add_patch(circle)
        else:
            print("Warning: Skip drawing bacteria with None radius!")


def save_snapshot(
    output_path,
    iteration,
    overwrite=False,
    formats=["png"],
):
    """
    Save an individual snapshots.

    Args:
        output_path(Path): Path of the stored results.
        iteration(int): Iteration count at which to obtain results.
        overwrite(bool): Enable to overwrite existing results. Disabling might speed up runtime when
        creating missing images.
        formats(list[str]): List of formats to store in. Only use formats supported by matplotlib.
    """
    quit = True
    for format in formats:
        save_path = _determine_image_save_path(output_path, iteration, fmt=format)

        # If the image is present we do not proceed unless the overwrite flag is active
        if overwrite or not os.path.isfile(save_path):
            quit = False

    if quit:
        return None

    # Get simulation settings and particles at the specified iteration
    domain, cells, meta_params = get_simulation_settings(output_path)
    df_cells = get_elements_at_iter(
        output_path,
        iteration,
        element_path="cell_storage",
    )
    df_voxels = get_elements_at_iter(
        output_path,
        iteration,
        element_path="voxel_storage",
    )

    fig, ax = _create_base_canvas(domain)

    mapper2 = _create_nutrient_voxel_color_mapping(domain)

    _plot_voxels(df_voxels, ax, mapper2)

    _plot_bacteria(df_cells, ax)

    # Plot labels in bottom left corner
    n_bacteria_1 = len(df_cells[df_cells["element.cell.cellular_reactions.species"] == "S1"])
    n_bacteria_2 = len(df_cells[df_cells["element.cell.cellular_reactions.species"] != "S1"])
    _plot_labels(fig, ax, n_bacteria_1, n_bacteria_2)

    # Save figure and cut off excess white space
    for format in formats:
        save_path = _determine_image_save_path(output_path, iteration, fmt=format)
        fig.savefig(
            save_path,
            bbox_inches="tight",
            pad_inches=0,
        )

    # Close the figure and free memory
    plt.close(fig)
    del df_cells
    del df_voxels
    del fig
    del ax
    gc.collect()


def __save_snapshot_helper(all_args):
    return save_snapshot(*all_args[0], **all_args[1])


def save_all_snapshots(
    output_path: Path,
    threads=1,
    show_bar=True,
    **kwargs,
):
    """
    Saves all snapshots using multiple threads.
    See :func:`save_snapshot` to store individual snapshots.

    Args:
        output_path(Path): Path of the stored results.
        threads(int): Number of threads to use simultaneously.
        show_bar(bool): Show or hide a progress bar.
        **kwargs: Any arguments for :func:`save_snapshot`
    """
    if threads <= 0:
        threads = os.cpu_count()
    all_args = [((output_path, iteration), kwargs) for iteration in get_all_iterations(output_path)]
    if show_bar:
        _ = list(
            tqdm.tqdm(
                mp.Pool(threads).imap_unordered(
                    __save_snapshot_helper,
                    all_args,
                ),
                total=len(all_args),
            )
        )
    else:
        mp.Pool(threads).imap_unordered(__save_snapshot_helper, all_args)


def save_movie(output_path):
    # Save snapshots and generate Movie
    save_all_snapshots(output_path, threads=-1)

    # Also create a movie with ffmpeg
    bashcmd = f"ffmpeg -v quiet -stats -y -r 30 -f image2 -pattern_type glob -i '{output_path}/images/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/movie_4.mp4"
    os.system(bashcmd)
