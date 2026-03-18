"""Finite-difference spatial operators, explicit time integrators, and CFL diagnostics.

This module provides the low-level numerical building blocks for solving the 1D viscous
Burgers equation: u_t + u * u_x = nu * u_xx. All spatial operators assume periodic
boundary conditions on a uniform grid and operate on plain NumPy arrays with no solver
state. The module offers two advection discretizations (upwind and Lax-Wendroff), a
standard central-difference diffusion operator, CFL-based adaptive time-step selection
that respects both the advective and diffusive stability limits, and two explicit time
integrators (forward Euler and Heun's RK2 method).
"""

import numpy as np


def upwind_advection_flux(u, dx):
    """Upwind finite difference for the nonlinear advection term u * u_x.

    Applies first-order flux-splitting on a periodic domain: where the local velocity
    u[i] > 0 the backward (left-biased) difference is used, and where u[i] < 0 the
    forward (right-biased) difference is used. This ensures information propagates in
    the correct characteristic direction and prevents non-physical oscillations near
    steep gradients and shocks. Periodic wrapping is handled via ``np.roll``.

    Args:
        u: solution array of shape (nx,).
        dx: uniform grid spacing.

    Returns:
        Array of shape (nx,) representing u * u_x at each grid point.
    """
    u_pos = np.maximum(u, 0.0)
    u_neg = np.minimum(u, 0.0)

    du_backward = (u - np.roll(u, 1)) / dx
    du_forward = (np.roll(u, -1) - u) / dx

    return u_pos * du_backward + u_neg * du_forward


def lax_wendroff_advection_flux(u, dx, dt):
    """Lax-Wendroff scheme for the nonlinear advection term (periodic BCs).

    A second-order accurate (in both space and time) discretization that uses a
    predictor-corrector approach. Half-step interface values are first computed via a
    Lax-Friedrichs-style average, then the corrected fluxes at cell interfaces yield
    the final spatial derivative. Uses the conservative flux form f(u) = u^2/2 of
    Burgers' equation. Higher accuracy than upwind, but may introduce dispersive
    (oscillatory) artifacts near discontinuities because it lacks built-in numerical
    dissipation.

    Args:
        u: solution array of shape (nx,).
        dx: uniform grid spacing.
        dt: current time step (needed for the predictor half-step).

    Returns:
        Array of shape (nx,) representing the Lax-Wendroff advection flux divergence.
    """
    u_right = np.roll(u, -1)
    u_left = np.roll(u, 1)

    # Flux function f(u) = u^2 / 2
    f = 0.5 * u**2
    f_right = 0.5 * u_right**2
    f_left = 0.5 * u_left**2

    # Standard Lax-Wendroff: half-step values at interfaces
    u_half_right = 0.5 * (u + u_right) - 0.5 * (dt / dx) * (f_right - f)
    u_half_left = 0.5 * (u_left + u) - 0.5 * (dt / dx) * (f - f_left)

    f_half_right = 0.5 * u_half_right**2
    f_half_left = 0.5 * u_half_left**2

    return (f_half_right - f_half_left) / dx


def central_diffusion(u, dx, nu):
    """Central second-order finite difference for the viscous diffusion term nu * u_xx.

    Applies the standard 3-point stencil: nu * (u[i+1] - 2*u[i] + u[i-1]) / dx^2 with
    periodic boundary conditions. Second-order accurate in space. Subject to the von
    Neumann stability constraint dt <= 0.5 * dx^2 / nu when combined with an explicit
    time integrator (enforced by ``compute_stable_dt``).

    Args:
        u: solution array of shape (nx,).
        dx: uniform grid spacing.
        nu: kinematic viscosity coefficient (must be >= 0).

    Returns:
        Array of shape (nx,) representing nu * u_xx at each grid point.
    """
    u_right = np.roll(u, -1)
    u_left = np.roll(u, 1)
    return nu * (u_right - 2.0 * u + u_left) / dx**2


def compute_cfl(u, dx, dt):
    """Compute the Courant-Friedrichs-Lewy (CFL) number: dt * max|u| / dx.

    The CFL number measures how far information travels (in grid cells) per time step.
    Values above 1.0 typically violate the stability limit for explicit schemes; the
    solver targets CFL ~ 0.5 for safety.

    Args:
        u: solution array of shape (nx,).
        dx: uniform grid spacing.
        dt: current time step.

    Returns:
        Scalar CFL number (dimensionless).
    """
    return dt * np.max(np.abs(u)) / dx


def compute_stable_dt(u, dx, nu, cfl_target=0.5, dt_max=1e-3):
    """Compute a stable time step satisfying both advection and diffusion constraints.

    Two independent stability limits are enforced:
      - Advection (CFL): dt <= cfl_target * dx / max(|u|, eps)
      - Diffusion (von Neumann): dt <= 0.5 * dx^2 / nu
    The returned dt is the minimum of these two limits and the user-specified cap
    ``dt_max``. The adaptive selection is critical: as the solution steepens and
    max|u| grows, the advective limit tightens automatically.

    Args:
        u: solution array of shape (nx,).
        dx: uniform grid spacing.
        nu: kinematic viscosity (> 0).
        cfl_target: target CFL number (default 0.5, conservative).
        dt_max: hard upper bound on dt.

    Returns:
        Stable time step size (scalar float).
    """
    u_max = np.max(np.abs(u))
    dt_advection = cfl_target * dx / max(u_max, 1e-10)
    dt_diffusion = 0.5 * dx**2 / nu if nu > 0 else np.inf
    return min(dt_advection, dt_diffusion, dt_max)


def rk2_step(u, rhs_fn, dt):
    """Advance one time step using Heun's method (explicit 2nd-order Runge-Kutta).

    Two-stage predictor-corrector scheme:
      k1 = rhs_fn(u)
      u_star = u + dt * k1           (Euler predictor)
      k2 = rhs_fn(u_star)
      u_new = u + 0.5 * dt * (k1 + k2)  (trapezoidal corrector)

    Second-order accurate in time (error ~ O(dt^2)), which complements the spatial
    accuracy of the finite-difference operators. This is the default time integrator.

    Args:
        u: current solution array.
        rhs_fn: callable that computes du/dt given u.
        dt: time step size.

    Returns:
        Updated solution array at time t + dt.
    """
    k1 = rhs_fn(u)
    u_star = u + dt * k1
    k2 = rhs_fn(u_star)
    return u + 0.5 * dt * (k1 + k2)


def euler_step(u, rhs_fn, dt):
    """Advance one time step using forward Euler (explicit 1st-order).

    Simplest explicit time integrator: u_new = u + dt * rhs_fn(u). Only first-order
    accurate (error ~ O(dt)), so it requires smaller time steps than RK2 for comparable
    accuracy. Provided as a baseline/fallback integrator.

    Args:
        u: current solution array.
        rhs_fn: callable that computes du/dt given u.
        dt: time step size.

    Returns:
        Updated solution array at time t + dt.
    """
    return u + dt * rhs_fn(u)
