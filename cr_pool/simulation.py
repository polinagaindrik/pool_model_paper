"""

.. list-table:: Recurring Arguments
    :header-rows: 1
    :widths: 15 10 75

    * - Name
      - Type
      - Description
    * - **output_path**
      - :class:`Path`
      - Path of the stored results.
    * - **iteration**
      - :class:`int`
      - Iteration count at which to obtain results.
    * - **element_path**
      - :class:`str`
      - Identifier which should be \"cell_storage\" or \"voxel_storage\"
    * - **threads**
      - :class:`int`
      - Number of threads to use simultaneously.
    * - **iteration**
      - :class:`int`
      - Iteration count at which to obtain results.

"""

import os
import json
from typing import Any
import pandas as pd
from pathlib import Path
from .cr_pool import *
import multiprocessing as mp
import numpy as np
import scipy as sp
from types import SimpleNamespace
import tqdm


def get_last_output_path(name="pool_model", prefix="out") -> Path:
    """
    Returns the path of the last numerical result.

    Args:
        name(str): Name of the subfolder in which to look for.
    Returns:
        Path: Storage Path of the last simulation result
    """
    return Path(prefix) / name / sorted(os.listdir(Path(prefix) / name))[-1]


def get_simulation_settings(output_path) -> tuple[Any, Any, Any]:
    """
    Obtain simulation settings for a given output path.

    Args:
        output_path(Path): Path of the stored results.
    Returns:
        (Any, Any, Any): A tuple containing json results of the domain, initial cells and meta
        parameters.
    """
    f_domain = open(output_path / "domain.json")
    f_initial_cells = open(output_path / "initial_cells.json")
    f_meta_params = open(output_path / "meta_params.json")

    domain = json.load(f_domain, object_hook=lambda d: SimpleNamespace(**d))
    initial_cells = json.load(f_initial_cells, object_hook=lambda d: SimpleNamespace(**d))
    meta_params = json.load(f_meta_params, object_hook=lambda d: SimpleNamespace(**d))
    return domain, initial_cells, meta_params


def _combine_batches(run_directory):
    # Opens all batches in a given directory and stores
    # them in one unified big list
    combined_batch = []
    for batch_file in os.listdir(run_directory):
        f = open(run_directory / batch_file)
        b = json.load(f)["data"]
        combined_batch.extend(b)
    return combined_batch


def _convert_entries(df, element_path):
    if element_path == "cell_storage":
        df["identifier.Division"] = df["identifier.Division"].apply(lambda x: tuple(x))
        df["element.identifier.Division"] = df["element.identifier.Division"].apply(
            lambda x: tuple(x)
        )
        df["element.cell.mechanics.pos"] = df["element.cell.mechanics.pos"].apply(
            lambda x: np.array(x)
        )
        df["element.cell.mechanics.vel"] = df["element.cell.mechanics.vel"].apply(
            lambda x: np.array(x)
        )
        df["element.cell.interactionextracellulargradient"] = df[
            "element.cell.interactionextracellulargradient"
        ].apply(lambda x: np.array(x))

    if element_path == "voxel_storage":
        df["element.index"] = df["element.index"].apply(lambda x: np.array(x))
        df["element.voxel.min"] = df["element.voxel.min"].apply(lambda x: np.array(x))
        df["element.voxel.max"] = df["element.voxel.max"].apply(lambda x: np.array(x))
        df["element.voxel.middle"] = df["element.voxel.middle"].apply(lambda x: np.array(x))
        df["element.voxel.dx"] = df["element.voxel.dx"].apply(lambda x: np.array(x))
        df["element.voxel.index"] = df["element.voxel.index"].apply(lambda x: np.array(x))
        df["element.voxel.extracellular_concentrations"] = df[
            "element.voxel.extracellular_concentrations"
        ].apply(lambda x: np.array(x))
        df["element.voxel.extracellular_gradient"] = df[
            "element.voxel.extracellular_gradient"
        ].apply(lambda x: np.array(x))
        df["element.voxel.diffusion_constant"] = df["element.voxel.diffusion_constant"].apply(
            lambda x: np.array(x)
        )
        df["element.voxel.production_rate"] = df["element.voxel.production_rate"].apply(
            lambda x: np.array(x)
        )
        df["element.voxel.degradation_rate"] = df["element.voxel.degradation_rate"].apply(
            lambda x: np.array(x)
        )
        df["element.neighbors"] = df["element.neighbors"].apply(lambda x: np.array(x))
        df["element.cells"] = df["element.cells"].apply(lambda x: np.array(x))
        df["element.new_cells"] = df["element.new_cells"].apply(lambda x: np.array(x))
        df["element.rng.seed"] = df["element.rng.seed"].apply(lambda x: np.array(x))
        df["element.extracellular_concentration_increments"] = df[
            "element.extracellular_concentration_increments"
        ].apply(lambda x: np.array(x))
        df["element.concentration_boundaries"] = df["element.concentration_boundaries"].apply(
            lambda x: np.array(x)
        )

    return df


def get_elements_at_iter(output_path: Path, iteration, element_path="cell_storage") -> pd.DataFrame:
    """
    Helper function to obtain information about cells or the domain at a given iteration point.

    Args:
        output_path(Path): Path of the stored results.
        iteration(int): Iteration count at which to obtain results.
        element_path(str): Identifier which should be \"cell_storage\" or \"voxel_storage\"
    Returns:
        pd.DataFrame: DataFrame containing all information
    """
    dir = Path(output_path) / element_path / "json"
    run_directory = None
    for x in os.listdir(dir):
        if int(x) == iteration:
            run_directory = dir / x
            break
    if run_directory is not None:
        df = pd.json_normalize(_combine_batches(run_directory))
        df = _convert_entries(df, element_path)
        return pd.DataFrame(df)
    else:
        raise ValueError(f"Could not find iteration {iteration} in saved results")


def get_all_iterations(output_path, element_path="cell_storage"):
    """
    Obtain a sorted list of all saved iteration counts.

    Args:
        output_path(Path): Path of the stored results.
        element_path(str): Identifier which should be \"cell_storage\" or \"voxel_storage\"
    Returns:
        list[int]: Sorted list of iteration counts.
    """
    return sorted([int(x) for x in os.listdir(Path(output_path) / element_path / "json")])


def __iter_to_elements(args):
    df = get_elements_at_iter(*args)
    df.insert(loc=0, column="iteration", value=args[1])
    return df


def get_elements_at_all_iterations(output_path: Path, element_path="cell_storage", threads=1):
    """
    Args:
        output_path(Path): Path of the stored results.
        element_path(str): Identifier which should be \"cell_storage\" or \"voxel_storage\"
        threads(int): Number of threads to use simultaneously.
    Returns:
    """
    if threads <= 0:
        threads = os.cpu_count()
    dir = Path(output_path) / element_path / "json"
    # runs = [(x, dir) for x in os.listdir(dir)]
    pool = mp.Pool(threads)
    all_iterations = get_all_iterations(output_path, element_path)
    result = list(
        tqdm.tqdm(
            pool.imap(
                __iter_to_elements,
                map(
                    lambda iteration: (output_path, iteration, element_path),
                    all_iterations,
                ),
            ),
            total=len(all_iterations),
        )
    )

    return pd.concat(result)


def calculate_spatial_density(data, domain, weights=None):
    """
    Calculates the spatial entry given data for cells and domain.

    Args:
        data(pd.DataFrame): DataFrame containing cellular properties.
        domain(pd.DataFrame): DataFrame containing domain properties.
        weights(list): List of weightings to use for calculating the entropy. Will be filled
        automatically by the volume of the individual cells if not provided.
    """
    positions = np.array([x for x in data["element.cell.mechanics.pos"]])

    if weights is None or weights is True:
        weights = (np.array(data["element.cell.cellular_reactions.cell_volume"]) / np.pi) ** 0.5

    return _calculate_spatial_density_from_positions(positions, domain, weights)


def _calculate_spatial_density_from_positions(positions, domain, weights=None):
    x = positions[:, 0]
    y = positions[:, 1]

    xmin = 0.0
    xmax = domain.size
    h = (BacteriaTemplate().cellular_reactions.cell_volume / np.pi) ** 0.5

    X, Y = np.mgrid[xmin:xmax:h, xmin:xmax:h]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = sp.stats.gaussian_kde(values, weights=weights)
    Z = np.reshape(kernel(positions).T, X.shape)
    return Z


def calculate_entropy(output_path, iteration) -> np.ndarray:
    """
    Obtain entropies for both species.

    Args:
        output_path(Path): Path of the stored results.
        iteration(int): Iteration count at which to obtain results.
    Returns:
        np.ndarray: Array containing entropy values for both species.
    """
    domain, _, _ = get_simulation_settings(output_path)
    data = get_elements_at_iter(output_path, iteration)

    data1 = data[data["element.cell.cellular_reactions.species"] == "S1"]
    data2 = data[data["element.cell.cellular_reactions.species"] != "S1"]

    Z1 = calculate_spatial_density(data1, domain, weights=True)
    Z2 = calculate_spatial_density(data2, domain, weights=True)

    z1 = sp.stats.entropy(np.rot90(Z1).reshape(-1))
    z2 = sp.stats.entropy(np.rot90(Z2).reshape(-1))

    # y = z1/(z1+z2)
    return np.array([z1, z2])
