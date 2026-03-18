import pytest
import numpy as np
import yaml


@pytest.fixture
def default_config():
    with open("configs/burgers_default.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def small_config():
    """Minimal config for fast tests."""
    return {
        "domain": {
            "x_min": 0.0,
            "x_max": 6.283185307179586,
            "nx_coarse": 32,
            "nx_fine": 128,
        },
        "physics": {"nu": 0.01},
        "time": {"t_end": 0.1, "cfl": 0.5, "dt_max": 0.001},
        "scheme": {
            "advection": "upwind",
            "diffusion": "central",
            "time_integrator": "rk2",
        },
        "initial_condition": {
            "type": "sine_sum",
            "params": {"n_modes": 2, "amplitude_range": [0.5, 1.0], "seed": 0},
        },
    }


@pytest.fixture
def coarse_grid():
    return np.linspace(0, 2 * np.pi, 64, endpoint=False)
