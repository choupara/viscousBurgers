"""Diverse initial condition (IC) families for the 1D Burgers equation.

Provides several analytically defined IC generators used to test solver behavior and ML
model generalization across different flow regimes. Three families are included:

- **sine_sum**: Smooth superposition of sinusoidal modes. Ideal for convergence-order
  testing because the solution remains smooth for moderate times.
- **gaussian_bump**: Localized Gaussian pulses that advect and diffuse, testing the
  interplay between nonlinear steepening and viscous smoothing.
- **riemann**: Piecewise-constant jump discontinuities that produce shocks (u_L > u_R)
  or rarefaction waves (u_L < u_R). Most demanding for coarse-grid solvers and for the
  ML correction model.

All ICs accept a ``seed`` parameter for reproducible random generation, allowing a
clear separation of training and held-out evaluation conditions.
"""

import numpy as np


def sine_sum(x, n_modes=3, amplitude_range=(0.5, 1.5), seed=42):
    """Sum of sinusoids on a periodic domain.

    Generates u(x) = sum_{k=1}^{n_modes} A_k * sin(2*pi*k*x/L + phi_k) where
    amplitudes A_k are drawn uniformly from ``amplitude_range`` and phases phi_k are
    uniform on [0, 2*pi). The smooth output is well-suited for verifying numerical
    convergence rates.

    Args:
        x: grid coordinate array of shape (nx,).
        n_modes: number of Fourier modes to superimpose.
        amplitude_range: (min, max) uniform range for mode amplitudes.
        seed: RNG seed for reproducible amplitude/phase selection.

    Returns:
        Initial condition array of shape (nx,).
    """
    rng = np.random.default_rng(seed)
    L = x[-1] - x[0] + (x[1] - x[0])  # full period length
    u = np.zeros_like(x)
    for k in range(1, n_modes + 1):
        amp = rng.uniform(amplitude_range[0], amplitude_range[1])
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(2 * np.pi * k * x / L + phase)
    return u


def gaussian_bump(x, n_bumps=1, width=0.1, seed=42):
    """Sum of localized Gaussian bumps on a periodic domain.

    Each bump is defined as A * exp(-(x - c)^2 / (2 * sigma^2)) with randomized
    center c, amplitude A, and sigma = width * L. Tests advection-diffusion interplay:
    the bump steepens under nonlinear advection while spreading due to viscosity.

    Args:
        x: grid coordinate array of shape (nx,).
        n_bumps: number of Gaussian pulses to superimpose.
        width: bump width as a fraction of the domain length.
        seed: RNG seed for reproducible center/amplitude selection.

    Returns:
        Initial condition array of shape (nx,).
    """
    rng = np.random.default_rng(seed)
    L = x[-1] - x[0] + (x[1] - x[0])
    u = np.zeros_like(x)
    for _ in range(n_bumps):
        center = rng.uniform(x[0], x[0] + L)
        amp = rng.uniform(0.5, 2.0)
        sigma = width * L
        u += amp * np.exp(-((x - center) ** 2) / (2 * sigma**2))
    return u


def riemann_problem(x, x_disc=None, u_left=1.0, u_right=0.0):
    """Piecewise-constant initial condition (Riemann problem).

    Creates a jump discontinuity at x_disc: u = u_left for x < x_disc, u = u_right
    otherwise. When u_left > u_right a shock wave forms; when u_left < u_right a
    rarefaction fan develops. This is the most challenging IC for coarse-grid schemes
    because the discontinuity demands high resolution that the coarse grid lacks,
    making it a strong test case for the ML correction model.

    Args:
        x: grid coordinate array of shape (nx,).
        x_disc: location of the discontinuity (defaults to domain midpoint).
        u_left: solution value to the left of the discontinuity.
        u_right: solution value to the right of the discontinuity.

    Returns:
        Initial condition array of shape (nx,).
    """
    L = x[-1] - x[0] + (x[1] - x[0])
    if x_disc is None:
        x_disc = x[0] + 0.5 * L
    u = np.where(x < x_disc, u_left, u_right)
    return u


def get_ic(ic_type, x, **kwargs):
    """Factory function dispatching to IC generators by name.

    Args:
        ic_type: one of "sine_sum", "gaussian_bump", or "riemann".
        x: grid coordinate array passed through to the chosen generator.
        **kwargs: additional keyword arguments forwarded to the generator.

    Returns:
        Initial condition array of shape (nx,).

    Raises:
        ValueError: if ``ic_type`` is not recognized.
    """
    ic_map = {
        "sine_sum": sine_sum,
        "gaussian_bump": gaussian_bump,
        "riemann": riemann_problem,
    }
    if ic_type not in ic_map:
        raise ValueError(f"Unknown IC type '{ic_type}'. Choose from {list(ic_map.keys())}")
    return ic_map[ic_type](x, **kwargs)
