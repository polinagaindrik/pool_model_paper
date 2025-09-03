import numpy as np
import scipy as sp

import cr_pool as crp


if __name__ == "__main__":
    # Domain Settings
    domain = crp.Domain()
    domain.size = 1_000
    domain.diffusion_constants = [5.0, 5.0]

    # Meta Parameters
    meta_params = crp.MetaParams()
    meta_params.save_interval = 1_000
    meta_params.n_times = 40_001
    meta_params.dt = 0.25
    meta_params.n_threads = 8

    cells = crp.generate_cells(18, 18, domain, randomness=0.0)

    output_path = crp.run_simulation(
        cells,
        domain,
        meta_params,
    )
