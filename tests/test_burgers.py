"""Tests for the BurgersSolver class."""

import numpy as np
import pytest

from src.pde.burgers import BurgersSolver
from src.pde.metrics import energy, l2_error, relative_l2_error, linf_error


class TestBurgersSolver:
    def test_solver_initialization(self, small_config):
        solver = BurgersSolver(small_config)
        assert solver.nx == 32
        assert solver.u.shape == (32,)
        assert solver.t == 0.0

    def test_nx_override(self, small_config):
        solver = BurgersSolver(small_config, nx_override=128)
        assert solver.nx == 128
        assert solver.u.shape == (128,)

    def test_solver_runs_without_crash(self, small_config):
        solver = BurgersSolver(small_config)
        history = solver.solve(t_end=0.05)
        assert solver.t == pytest.approx(0.05, abs=1e-10)
        assert len(history) >= 2  # at least initial + final

    def test_energy_decreases_viscous(self, small_config):
        """For viscous Burgers, energy must monotonically decrease."""
        small_config["time"]["t_end"] = 0.2
        solver = BurgersSolver(small_config)
        history = solver.solve(t_end=0.2, record_interval=1)

        dx = solver.dx
        energies = [energy(u, dx) for _, u in history]

        for i in range(1, len(energies)):
            assert energies[i] <= energies[i - 1] + 1e-10, (
                f"Energy increased at step {i}: {energies[i-1]:.6e} -> {energies[i]:.6e}"
            )

    def test_solution_stays_finite(self, small_config):
        solver = BurgersSolver(small_config)
        solver.solve(t_end=0.1)
        assert np.all(np.isfinite(solver.u))

    def test_correction_fn_is_called(self, small_config):
        """Verify the correction_fn hook works."""
        corrections_applied = []

        def dummy_correction(u, x, dx, dt, nu):
            corrections_applied.append(1)
            return np.zeros_like(u)

        solver = BurgersSolver(small_config)
        solver.solve(t_end=0.01, correction_fn=dummy_correction)
        assert len(corrections_applied) > 0

    def test_lax_wendroff_scheme(self, small_config):
        small_config["scheme"]["advection"] = "lax_wendroff"
        solver = BurgersSolver(small_config)
        solver.solve(t_end=0.05)
        assert np.all(np.isfinite(solver.u))

    def test_euler_integrator(self, small_config):
        small_config["scheme"]["time_integrator"] = "euler"
        solver = BurgersSolver(small_config)
        solver.solve(t_end=0.05)
        assert np.all(np.isfinite(solver.u))


class TestMetrics:
    def test_l2_error_zero_for_identical(self):
        u = np.array([1.0, 2.0, 3.0])
        assert l2_error(u, u, dx=0.1) == pytest.approx(0.0)

    def test_l2_error_positive(self):
        u = np.array([1.0, 2.0])
        u_ref = np.array([1.1, 2.1])
        assert l2_error(u, u_ref, dx=0.1) > 0

    def test_relative_l2_error_bounded(self):
        u = np.array([1.0, 2.0])
        u_ref = np.array([1.0, 2.0])
        assert relative_l2_error(u, u_ref, dx=0.1) == pytest.approx(0.0)

    def test_linf_error(self):
        u = np.array([1.0, 2.0, 3.0])
        u_ref = np.array([1.1, 1.5, 3.0])
        assert linf_error(u, u_ref) == pytest.approx(0.5)

    def test_energy_positive(self):
        u = np.array([1.0, -2.0, 3.0])
        assert energy(u, dx=0.1) > 0
