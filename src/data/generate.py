"""Paired coarse/fine simulation runner for generating ML correction targets.

This module is the first stage of the data pipeline. For every (IC family, seed, nu)
combination it runs *two* independent Burgers simulations on the same physical problem:

  1. A **fine-grid** solver (e.g., 256 points) that serves as the ground-truth reference.
  2. A **coarse-grid** solver (e.g., 64 points) that has higher discretization error.

Both solvers are advanced to the same physical snapshot times. At each snapshot the fine
solution is linearly interpolated onto the coarse grid, and the correction target is
computed:

    Delta_u = u_fine_aligned - u_coarse

This Delta_u is what the ML model will learn to predict. It represents the additional
information that the fine grid resolves but the coarse grid misses — primarily near
steep gradients, shocks, and regions of strong advection-diffusion interplay.

Results are saved as compressed ``.npz`` files, one per simulation, containing the
coarse/fine grids, time snapshots, and correction targets.

Usage:
    python -m src.data.generate --config configs/dataset_gen.yaml
"""

import os
import argparse

import numpy as np
import yaml
from scipy.interpolate import interp1d

from src.pde.burgers import BurgersSolver


def run_paired_simulation(config, ic_type, ic_seed, nu):
    """Run one paired coarse/fine simulation and compute correction targets.

    Both solvers share the same physical parameters and initial condition but operate on
    grids of different resolution. The fine solution is interpolated onto the coarse grid
    at each snapshot time, and the pointwise difference (Delta_u) gives the correction
    that the ML model will learn.

    Args:
        config: solver config dict (from burgers_default.yaml with overrides).
        ic_type: initial condition family name ("sine_sum", "gaussian_bump", "riemann").
        ic_seed: random seed for IC generation (controls reproducibility).
        nu: kinematic viscosity for this run.

    Returns:
        Dict with keys: x_coarse, x_fine, nu, ic_type, ic_seed, snapshots. Each
        snapshot contains t, u_coarse, u_fine_aligned, delta_u, and dt_coarse.
    """
    # Override IC and viscosity in config
    cfg = _make_config(config, ic_type, ic_seed, nu)
    nx_coarse = cfg["domain"]["nx_coarse"]
    nx_fine = cfg["domain"]["nx_fine"]

    # Create solvers
    solver_coarse = BurgersSolver(cfg, nx_override=nx_coarse)
    solver_fine = BurgersSolver(cfg, nx_override=nx_fine)

    t_end = cfg["time"]["t_end"]
    n_snapshots = cfg.get("n_snapshots_per_sim", 20)
    snapshot_times = np.linspace(0, t_end, n_snapshots + 1)[1:]  # exclude t=0

    snapshots = []
    for t_target in snapshot_times:
        # Advance both solvers to the target time
        _advance_to(solver_coarse, t_target)
        _advance_to(solver_fine, t_target)

        # Interpolate fine solution to coarse grid
        interp_fn = interp1d(
            solver_fine.x, solver_fine.u, kind="linear", assume_sorted=True
        )
        u_fine_aligned = interp_fn(solver_coarse.x)

        delta_u = u_fine_aligned - solver_coarse.u

        snapshots.append({
            "t": solver_coarse.t,
            "u_coarse": solver_coarse.u.copy(),
            "u_fine_aligned": u_fine_aligned,
            "delta_u": delta_u,
            "dt_coarse": solver_coarse.dx,  # last dx used (for features)
        })

    return {
        "x_coarse": solver_coarse.x,
        "x_fine": solver_fine.x,
        "nu": nu,
        "ic_type": ic_type,
        "ic_seed": ic_seed,
        "snapshots": snapshots,
    }


def generate_dataset(config_path):
    """Generate the full paired-simulation dataset.

    Reads ``dataset_gen.yaml`` and the solver defaults, then iterates over all
    (ic_family, seed, nu) combinations to produce paired coarse/fine simulations. Each
    result is saved as a compressed ``.npz`` file named
    ``{ic_type}_seed{seed}_nu{nu:.4f}.npz`` in the configured output directory.

    Args:
        config_path: path to dataset_gen.yaml.
    """
    with open(config_path) as f:
        gen_cfg = yaml.safe_load(f)

    with open("configs/burgers_default.yaml") as f:
        solver_cfg = yaml.safe_load(f)

    # Override refinement ratio
    ratio = gen_cfg["refinement_ratio"]
    nx_coarse = solver_cfg["domain"]["nx_coarse"]
    solver_cfg["domain"]["nx_fine"] = nx_coarse * ratio

    output_dir = gen_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    ic_families = gen_cfg["ic_families"]
    ic_seeds = gen_cfg["ic_seeds"]
    nu_min, nu_max = gen_cfg["nu_range"]
    n_nu = max(1, gen_cfg.get("n_samples", 500) // (len(ic_families) * len(ic_seeds)))
    nu_values = np.linspace(nu_min, nu_max, min(n_nu, 10))

    solver_cfg["n_snapshots_per_sim"] = gen_cfg["n_snapshots_per_sim"]

    count = 0
    for ic_type in ic_families:
        for seed in ic_seeds:
            for nu in nu_values:
                result = run_paired_simulation(solver_cfg, ic_type, seed, nu)

                fname = f"{ic_type}_seed{seed}_nu{nu:.4f}.npz"
                _save_result(os.path.join(output_dir, fname), result)
                count += 1

                if count % 10 == 0:
                    print(f"  Generated {count} simulations...")

    print(f"Dataset generation complete: {count} simulations saved to {output_dir}/")


def _make_config(config, ic_type, ic_seed, nu):
    """Create a modified config for a specific simulation."""
    # Only carry over default IC params if the IC type matches the default;
    # otherwise different IC families receive incompatible keyword arguments.
    default_ic = config["initial_condition"]
    if ic_type == default_ic.get("type"):
        ic_params = dict(default_ic.get("params", {}))
    else:
        ic_params = {}

    cfg = {
        "domain": dict(config["domain"]),
        "physics": {"nu": nu},
        "time": dict(config["time"]),
        "scheme": dict(config["scheme"]),
        "initial_condition": {
            "type": ic_type,
            "params": ic_params,
        },
    }
    if ic_type != "riemann":
        cfg["initial_condition"]["params"]["seed"] = ic_seed
    if "n_snapshots_per_sim" in config:
        cfg["n_snapshots_per_sim"] = config["n_snapshots_per_sim"]
    return cfg


def _advance_to(solver, t_target):
    """Advance a solver to exactly t_target using adaptive time stepping.

    Uses the solver's own CFL-based dt selection, clamping the final step so that the
    simulation lands precisely on the target time. This ensures coarse and fine solvers
    are synchronized to the same physical snapshots despite taking different numbers of
    time steps (due to different dx and therefore different stable dt values).
    """
    from src.pde.schemes import compute_stable_dt, rk2_step, euler_step

    while solver.t < t_target - 1e-14:
        dt = compute_stable_dt(
            solver.u, solver.dx, solver.nu, solver.cfl_target, solver.dt_max
        )
        dt = min(dt, t_target - solver.t)

        if solver.time_integrator == "rk2":
            solver.u = rk2_step(solver.u, solver.rhs, dt)
        elif solver.time_integrator == "euler":
            solver.u = euler_step(solver.u, solver.rhs, dt)

        solver.t += dt


def _save_result(path, result):
    """Save a paired-simulation result dict as a compressed .npz archive.

    Flattens the nested snapshot structure into keyed arrays (e.g., u_coarse_0,
    delta_u_0, t_0, ...) so that NumPy can serialize everything efficiently.
    """
    flat = {
        "x_coarse": result["x_coarse"],
        "x_fine": result["x_fine"],
        "nu": np.array([result["nu"]]),
        "ic_seed": np.array([result["ic_seed"]]),
    }
    for i, snap in enumerate(result["snapshots"]):
        flat[f"t_{i}"] = np.array([snap["t"]])
        flat[f"u_coarse_{i}"] = snap["u_coarse"]
        flat[f"u_fine_aligned_{i}"] = snap["u_fine_aligned"]
        flat[f"delta_u_{i}"] = snap["delta_u"]

    flat["n_snapshots"] = np.array([len(result["snapshots"])])
    flat["ic_type"] = np.array([result["ic_type"]])
    np.savez_compressed(path, **flat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paired simulation dataset")
    parser.add_argument("--config", default="configs/dataset_gen.yaml")
    args = parser.parse_args()
    generate_dataset(args.config)
