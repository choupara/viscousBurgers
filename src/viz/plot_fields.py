"""Field snapshot visualization: solution profiles at a single time instant.

Provides two diagnostic plots:
- **Field comparison**: Overlays the coarse, corrected, and fine reference solutions on
  a single axes to visually assess how well the correction closes the gap.
- **Correction field**: Compares the predicted correction Delta_u against the true
  correction, with a residual panel showing where the model under- or over-predicts.

Uses the non-interactive Agg backend so figures can be generated in headless
environments (CI, remote servers) and saved directly to disk.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_field_comparison(x, u_coarse, u_corrected, u_fine, t, nu, save_path=None):
    """Plot three-line comparison of coarse, corrected, and fine reference solutions.

    The fine reference (solid black) is the ground truth, the coarse solution (blue
    dashed) shows the uncorrected discretization error, and the corrected solution (red
    solid) shows the ML-augmented result. Ideally the red line should closely track the
    black line.

    Args:
        x: grid coordinate array of shape (nx,).
        u_coarse: coarse solver solution array (uncorrected).
        u_corrected: ML-corrected solver solution array.
        u_fine: fine reference solution (interpolated to coarse grid).
        t: simulation time at this snapshot.
        nu: viscosity (shown in the title for context).
        save_path: if provided, saves the figure to this file path (PNG or PDF).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, u_fine, "k-", linewidth=1.5, label="Fine (reference)")
    ax.plot(x, u_coarse, "b--", linewidth=1.2, label="Coarse")
    ax.plot(x, u_corrected, "r-", linewidth=1.2, label="Coarse + correction")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(f"t = {t:.3f}, nu = {nu}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_correction_field(x, delta_u_true, delta_u_pred, t, save_path=None):
    """Compare predicted vs true correction fields with a residual panel.

    Top panel overlays the ground-truth Delta_u (black) with the model prediction (red
    dashed). Bottom panel shows the pointwise residual (true - predicted), highlighting
    regions where the model under- or over-predicts the correction.

    Args:
        x: grid coordinate array of shape (nx,).
        delta_u_true: ground truth correction array (u_fine_aligned - u_coarse).
        delta_u_pred: model-predicted correction array.
        t: simulation time at this snapshot.
        save_path: if provided, saves the figure to this file path.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(x, delta_u_true, "k-", label="True correction")
    axes[0].plot(x, delta_u_pred, "r--", label="Predicted correction")
    axes[0].set_ylabel("delta_u")
    axes[0].set_title(f"Correction field at t = {t:.3f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, delta_u_true - delta_u_pred, "g-", linewidth=0.8)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Correction error")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
