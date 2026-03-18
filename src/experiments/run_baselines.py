"""Baseline experiment runner: quantifies coarse-grid discretization error.

Runs the coarse solver (with both upwind and Lax-Wendroff advection schemes) against a
fine-grid reference across all IC families and several viscosity values. The resulting
error trajectories (relative L2, L-infinity) establish the *uncorrected* performance
baseline that the ML correction model must improve upon. Results are saved as JSON for
comparison plots and tables.

This is a necessary first step before evaluating the correction model — it answers the
question "how much error does the coarse grid introduce?" and provides the denominator
for computing improvement ratios.
"""

import os
import argparse
import json

import numpy as np
import yaml

from src.pde.burgers import BurgersSolver
from src.pde.metrics import l2_error, relative_l2_error, linf_error, energy
from src.data.generate import _advance_to

from scipy.interpolate import interp1d


def run_baselines(config_path="configs/burgers_default.yaml", output_dir="outputs/baselines"):
    """Run all baseline coarse solver configurations and record error metrics.

    For each combination of (IC family, viscosity, advection scheme), runs paired
    coarse/fine simulations and records relative L2 and L-infinity errors at evenly
    spaced snapshot times. The fine-grid solution (interpolated to the coarse grid)
    serves as the reference truth.

    Args:
        config_path: path to the solver default configuration YAML.
        output_dir: directory where baseline_results.json will be written.

    Returns:
        List of result dicts, one per (ic_type, nu, scheme) combination.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)

    ic_families = ["sine_sum", "gaussian_bump", "riemann"]
    nu_values = [0.005, 0.01, 0.05]
    schemes = ["upwind", "lax_wendroff"]
    n_snapshots = 10
    ic_params_map = {
    "sine_sum": {"n_modes": 3, "amplitude_range": [0.5, 1.5], "seed": 42},
    "gaussian_bump": {"n_bumps": 1, "width": 0.1, "seed": 42},
    "riemann": {},
    }


    results = []

    for ic_type in ic_families:
        for nu in nu_values:
            for scheme in schemes:
                test_cfg = dict(cfg)
                test_cfg["physics"] = {"nu": nu}
                test_cfg["scheme"] = dict(cfg["scheme"])
                test_cfg["scheme"]["advection"] = scheme
                test_cfg["initial_condition"] = {
                    "type": ic_type,
                    "params": ic_params_map[ic_type],
                }

                t_end = cfg["time"]["t_end"]
                snap_times = np.linspace(0, t_end, n_snapshots + 1)[1:]

                solver_fine = BurgersSolver(test_cfg, nx_override=cfg["domain"]["nx_fine"])
                solver_coarse = BurgersSolver(test_cfg)

                errors_l2 = []
                errors_linf = []

                for t_target in snap_times:
                    _advance_to(solver_fine, t_target)
                    _advance_to(solver_coarse, t_target)

                    interp_fn = interp1d(
                        solver_fine.x, solver_fine.u, kind="linear", assume_sorted=True
                    )
                    u_fine_on_coarse = interp_fn(solver_coarse.x)

                    dx = solver_coarse.dx
                    errors_l2.append(relative_l2_error(solver_coarse.u, u_fine_on_coarse, dx))
                    errors_linf.append(linf_error(solver_coarse.u, u_fine_on_coarse))

                result = {
                    "ic_type": ic_type,
                    "nu": nu,
                    "scheme": scheme,
                    "times": snap_times.tolist(),
                    "relative_l2_errors": errors_l2,
                    "linf_errors": errors_linf,
                    "final_l2_error": errors_l2[-1],
                    "final_linf_error": errors_linf[-1],
                }
                results.append(result)
                print(f"{ic_type:15s} | nu={nu:.3f} | {scheme:15s} | L2={errors_l2[-1]:.4f} | Linf={errors_linf[-1]:.4f}")

    # Save results
    with open(os.path.join(output_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline results saved to {output_dir}/baseline_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--config", default="configs/burgers_default.yaml")
    parser.add_argument("--output", default="outputs/baselines")
    args = parser.parse_args()
    run_baselines(args.config, args.output)
