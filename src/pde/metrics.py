"""Error metrics and energy diagnostics for comparing numerical solutions.

Provides discrete-norm error measures (L2, relative L2, L-infinity) for quantifying
the accuracy of a coarse or ML-corrected solution against a fine-grid reference, plus
energy diagnostics for monitoring physical plausibility. For the viscous Burgers
equation the discrete kinetic energy E = 0.5 * dx * sum(u^2) must decrease
monotonically in time due to viscous dissipation — a violation signals numerical
instability. These metrics are used both in unit tests (e.g., convergence order
verification) and in experiment evaluation (one-step and rollout comparisons).
"""

import numpy as np


def l2_error(u, u_ref, dx):
    """Discrete L2 norm of the error: sqrt(dx * sum((u - u_ref)^2)).

    Approximates the continuous L2 integral via the trapezoidal rule on a uniform grid.

    Args:
        u: computed solution array of shape (nx,).
        u_ref: reference solution array of shape (nx,) (same grid).
        dx: uniform grid spacing.

    Returns:
        Scalar L2 error (>= 0).
    """
    return np.sqrt(dx * np.sum((u - u_ref) ** 2))


def relative_l2_error(u, u_ref, dx):
    """L2 error normalized by the L2 norm of u_ref.

    Dimensionless measure of error magnitude relative to the reference signal strength.
    Returns 0.0 when the reference norm is near zero (trivial solution).

    Args:
        u: computed solution array of shape (nx,).
        u_ref: reference solution array of shape (nx,).
        dx: uniform grid spacing.

    Returns:
        Relative L2 error in [0, inf). A value of 0.10 means 10% relative error.
    """
    ref_norm = np.sqrt(dx * np.sum(u_ref**2))
    if ref_norm < 1e-15:
        return 0.0
    return l2_error(u, u_ref, dx) / ref_norm


def linf_error(u, u_ref):
    """Maximum pointwise absolute error: max|u - u_ref|.

    Captures the worst-case error across the grid, typically located near steep
    gradients or shocks where discretization error is largest.

    Args:
        u: computed solution array of shape (nx,).
        u_ref: reference solution array of shape (nx,).

    Returns:
        Scalar L-infinity error (>= 0).
    """
    return np.max(np.abs(u - u_ref))


def energy(u, dx):
    """Discrete kinetic energy: E = 0.5 * dx * sum(u^2).

    For the viscous Burgers equation, energy must decrease monotonically in time due to
    viscous dissipation (dE/dt <= 0). An increase signals numerical instability. Used
    in unit tests and rollout evaluation.

    Args:
        u: solution array of shape (nx,).
        dx: uniform grid spacing.

    Returns:
        Scalar energy value (> 0 for non-trivial solutions).
    """
    return 0.5 * dx * np.sum(u**2)


def energy_drift(energy_series, energy_0):
    """Relative energy change over time: (E(t) - E(0)) / E(0).

    Tracks how far the energy deviates from its initial value as a fraction. For a
    well-behaved viscous simulation, all values should be negative (energy lost to
    dissipation). Positive values indicate non-physical energy growth.

    Args:
        energy_series: list or array of energy values at successive times.
        energy_0: initial energy E(0).

    Returns:
        Array of relative energy deviations, same length as ``energy_series``.
    """
    energy_series = np.asarray(energy_series)
    if abs(energy_0) < 1e-15:
        return np.zeros_like(energy_series)
    return (energy_series - energy_0) / energy_0
