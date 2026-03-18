"""Error trajectory and energy drift visualization.

Produces two key diagnostic plots for evaluating correction model quality:

- **Error vs time**: Semi-log plot of the relative L2 error over the simulation
  horizon for both the uncorrected coarse solver and the ML-corrected solver. If the
  correction is working, the red line (corrected) should sit below the blue line
  (coarse) at every snapshot time.
- **Energy evolution**: Tracks discrete kinetic energy over time for all three solver
  variants (fine, coarse, corrected). For the viscous Burgers equation energy should
  decrease monotonically. Energy growth in the corrected solver signals that the ML
  correction is injecting non-physical energy — a stability failure.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_error_vs_time(times, errors_coarse, errors_corrected, metric_name="Relative L2", save_path=None):
    """Semi-log plot of error trajectories for coarse and corrected solvers.

    The y-axis uses a logarithmic scale to make it easy to see order-of-magnitude
    improvements from the correction. Marker styles distinguish the two curves.

    Args:
        times: list of snapshot times (x-axis values).
        errors_coarse: error values at each time for the uncorrected coarse solver.
        errors_corrected: error values at each time for the ML-corrected solver.
        metric_name: label for the y-axis (e.g., "Relative L2", "L-infinity").
        save_path: if provided, saves the figure to this file path.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogy(times, errors_coarse, "b-o", markersize=4, label="Coarse")
    ax.semilogy(times, errors_corrected, "r-s", markersize=4, label="Coarse + correction")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{metric_name} Error")
    ax.set_title(f"{metric_name} Error vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_energy_drift(times, energy_coarse, energy_corrected, energy_fine, save_path=None):
    """Plot energy evolution for all three solver variants.

    For the viscous Burgers equation, energy E = 0.5 * dx * sum(u^2) must decrease
    monotonically due to viscous dissipation. Comparing the three curves reveals whether
    the correction preserves this physical invariant. The fine reference (black) is the
    ground truth; the corrected solver (red) should track it more closely than the
    uncorrected coarse solver (blue dashed).

    Args:
        times: list of snapshot times (x-axis values).
        energy_coarse: energy trajectory for the uncorrected coarse solver.
        energy_corrected: energy trajectory for the ML-corrected solver.
        energy_fine: energy trajectory for the fine reference solver.
        save_path: if provided, saves the figure to this file path.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(times, energy_fine, "k-", linewidth=1.5, label="Fine (reference)")
    ax.plot(times, energy_coarse, "b--", linewidth=1.2, label="Coarse")
    ax.plot(times, energy_corrected, "r-", linewidth=1.2, label="Coarse + correction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
