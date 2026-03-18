"""1D viscous Burgers equation solver with optional ML correction hook.

Central orchestrator that wires together the spatial operators from ``schemes.py`` and
the initial condition generators from ``initial_conditions.py`` into a complete
time-stepping solver for:

    u_t + u * u_x = nu * u_xx      (periodic domain, x in [x_min, x_max))

The solver supports two advection schemes (upwind, Lax-Wendroff), two time integrators
(forward Euler, RK2), and adaptive CFL-based time stepping. The key design feature is
the ``correction_fn`` hook in the ``solve()`` method: when set to ``None`` the solver
runs as a standard numerical PDE solver, but when a trained ML model is plugged in via
this hook the solver applies a learned correction after every time step, compensating
for coarse-grid discretization error. This single hook is the integration point between
the classical numerics and the machine-learning pipeline.
"""

import numpy as np

from .schemes import (
    upwind_advection_flux,
    lax_wendroff_advection_flux,
    central_diffusion,
    compute_stable_dt,
    rk2_step,
    euler_step,
)
from .initial_conditions import get_ic


class BurgersSolver:
    """1D viscous Burgers equation solver.

    Solves u_t + u * u_x = nu * u_xx on a periodic domain with configurable advection
    scheme, time integrator, and adaptive CFL-based time stepping. The solver maintains
    a solution history for post-hoc analysis and visualization.

    The ``correction_fn`` hook (accepted by ``solve()``) is the key ML integration
    point: after each time step the hook receives the current state (u, x, dx, dt, nu)
    and returns an additive correction array that is applied to the solution. During
    baseline (classical) runs this hook is ``None``; during corrected runs it wraps the
    trained ML model.

    Attributes:
        nx: number of grid points.
        dx: uniform grid spacing.
        x: grid coordinate array of shape (nx,).
        nu: kinematic viscosity coefficient.
        u: current solution array of shape (nx,).
        t: current simulation time.
        history: list of (t, u.copy()) snapshot tuples.
    """

    def __init__(self, config, nx_override=None):
        """Initialize from a config dict.

        Args:
            config: dict loaded from burgers_default.yaml
            nx_override: optional grid size override (for fine/coarse runs)
        """
        domain = config["domain"]
        self.x_min = domain["x_min"]
        self.x_max = domain["x_max"]
        self.nx = nx_override if nx_override is not None else domain["nx_coarse"]
        self.L = self.x_max - self.x_min
        self.dx = self.L / self.nx
        self.x = np.linspace(self.x_min, self.x_min + self.L, self.nx, endpoint=False)

        self.nu = config["physics"]["nu"]

        time_cfg = config["time"]
        self.cfl_target = time_cfg["cfl"]
        self.dt_max = time_cfg["dt_max"]

        self.advection_scheme = config["scheme"]["advection"]
        self.time_integrator = config["scheme"]["time_integrator"]

        # Initialize solution
        ic_cfg = config["initial_condition"]
        self.u = get_ic(ic_cfg["type"], self.x, **ic_cfg.get("params", {}))
        self.t = 0.0
        self.history = [(0.0, self.u.copy())]

    def rhs(self, u):
        """Compute the right-hand side du/dt = -advection + diffusion.

        Assembles the spatial discretization by combining the selected advection scheme
        (upwind or Lax-Wendroff) with the central-difference diffusion operator. The
        result is passed to the time integrator (RK2 or Euler) during ``step()`` and
        ``solve()``.

        Args:
            u: current solution array of shape (nx,).

        Returns:
            Array of shape (nx,) representing du/dt.
        """
        if self.advection_scheme == "upwind":
            advection = upwind_advection_flux(u, self.dx)
        elif self.advection_scheme == "lax_wendroff":
            dt = compute_stable_dt(u, self.dx, self.nu, self.cfl_target, self.dt_max)
            advection = lax_wendroff_advection_flux(u, self.dx, dt)
        else:
            raise ValueError(f"Unknown advection scheme: {self.advection_scheme}")

        diffusion = central_diffusion(u, self.dx, self.nu)
        return -advection + diffusion

    def step(self, correction=None):
        """Advance the solution by one adaptively-chosen time step.

        Computes a stable dt via CFL/diffusion constraints, applies the configured time
        integrator, and optionally adds an external correction array (used during ML-
        corrected runs).

        Args:
            correction: optional array of shape (nx,) added to u after the PDE step.

        Returns:
            The time step dt that was actually used (float).
        """
        dt = compute_stable_dt(self.u, self.dx, self.nu, self.cfl_target, self.dt_max)

        if self.time_integrator == "rk2":
            self.u = rk2_step(self.u, self.rhs, dt)
        elif self.time_integrator == "euler":
            self.u = euler_step(self.u, self.rhs, dt)
        else:
            raise ValueError(f"Unknown time integrator: {self.time_integrator}")

        if correction is not None:
            self.u += correction

        self.t += dt
        return dt

    def solve(self, t_end, record_interval=10, correction_fn=None):
        """Integrate the Burgers equation forward in time until t >= t_end.

        Uses adaptive time stepping with the configured CFL target. The final step is
        clamped so that the solver lands exactly on ``t_end``. An optional
        ``correction_fn`` is called after every time step, receives the current state,
        and returns an additive correction that is applied to the solution. This is the
        main entry point for both baseline and ML-corrected simulation runs.

        Args:
            t_end: target final simulation time.
            record_interval: save a snapshot every N time steps.
            correction_fn: optional callable(u, x, dx, dt, nu) -> correction array.
                When None the solver runs as a standard (uncorrected) PDE solver.

        Returns:
            List of (t, u) snapshot tuples including the initial and final states.
        """
        step_count = 0
        while self.t < t_end - 1e-14:
            # Compute dt, possibly clamped to not overshoot t_end
            dt = compute_stable_dt(
                self.u, self.dx, self.nu, self.cfl_target, self.dt_max
            )
            dt = min(dt, t_end - self.t)

            if self.time_integrator == "rk2":
                self.u = rk2_step(self.u, self.rhs, dt)
            elif self.time_integrator == "euler":
                self.u = euler_step(self.u, self.rhs, dt)

            self.t += dt
            step_count += 1

            # Apply ML correction if provided
            if correction_fn is not None:
                correction = correction_fn(self.u, self.x, self.dx, dt, self.nu)
                self.u += correction

            # Halt early if solution has become non-finite
            if not np.all(np.isfinite(self.u)):
                import warnings
                warnings.warn(
                    f"Non-finite values detected at t={self.t:.6f} "
                    f"(step {step_count}). Stopping integration early."
                )
                break

            if step_count % record_interval == 0:
                self.history.append((self.t, self.u.copy()))

        # Always record final state
        if len(self.history) == 0 or abs(self.history[-1][0] - self.t) > 1e-14:
            self.history.append((self.t, self.u.copy()))

        return self.history

    def get_state(self):
        """Return a serializable snapshot of the current solver state.

        Returns:
            Dict with keys: t, u, x, nx, dx, nu (arrays are copied).
        """
        return {
            "t": self.t,
            "u": self.u.copy(),
            "x": self.x.copy(),
            "nx": self.nx,
            "dx": self.dx,
            "nu": self.nu,
        }
