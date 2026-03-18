"""Batch figure regeneration from saved experiment results.

Reads JSON result files produced by ``run_baselines.py`` and ``run_with_correction.py``
and regenerates all diagnostic plots (error trajectories, baseline comparisons). This
ensures figures are always reproducible from the raw experiment data without re-running
the (potentially expensive) simulations. Called as the final step of
``scripts/reproduce_all.sh``.
"""

import os
import json

from src.viz.plot_fields import plot_field_comparison
from src.viz.plot_errors import plot_error_vs_time, plot_energy_drift


def make_all_figures(results_dir="outputs", output_dir="outputs/figures"):
    """Regenerate all figures from saved experiment JSON files.

    Scans for correction_results.json and baseline_results.json in the results
    directory and produces corresponding plots. Silently skips any missing result files.

    Args:
        results_dir: root directory containing experiment output subdirectories
            (baselines/, corrected/).
        output_dir: directory where generated figure files (PNG) will be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot error trajectories from correction results
    corrected_path = os.path.join(results_dir, "corrected", "correction_results.json")
    if os.path.exists(corrected_path):
        with open(corrected_path) as f:
            results = json.load(f)

        for r in results:
            ic = r["ic_type"]
            seed = r.get("seed", 0)
            plot_error_vs_time(
                r["times"],
                r["coarse_errors"],
                r["corrected_errors"],
                metric_name="Relative L2",
                save_path=os.path.join(output_dir, f"error_{ic}_seed{seed}.png"),
            )
        print(f"Generated error trajectory plots in {output_dir}/")

    # Plot baseline results
    baseline_path = os.path.join(results_dir, "baselines", "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            results = json.load(f)

        for r in results:
            ic = r["ic_type"]
            nu = r["nu"]
            scheme = r["scheme"]
            plot_error_vs_time(
                r["times"],
                r["relative_l2_errors"],
                r["relative_l2_errors"],  # same for baseline (no correction)
                metric_name=f"Baseline L2 ({scheme})",
                save_path=os.path.join(
                    output_dir, f"baseline_{ic}_nu{nu:.3f}_{scheme}.png"
                ),
            )
        print(f"Generated baseline plots in {output_dir}/")

    if not os.path.exists(corrected_path) and not os.path.exists(baseline_path):
        print("No experiment results found. Run experiments first.")


if __name__ == "__main__":
    make_all_figures()
