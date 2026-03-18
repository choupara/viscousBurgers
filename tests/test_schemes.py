"""Tests for finite-difference schemes and time integrators."""

import numpy as np
import pytest

from src.pde.schemes import (
    upwind_advection_flux,
    lax_wendroff_advection_flux,
    central_diffusion,
    compute_cfl,
    compute_stable_dt,
    rk2_step,
    euler_step,
)


class TestDiffusion:
    """Test central diffusion against analytical heat equation decay."""

    def test_diffusion_operator_shape(self):
        u = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
        result = central_diffusion(u, dx=2 * np.pi / 64, nu=0.01)
        assert result.shape == u.shape

    def test_diffusion_of_constant_is_zero(self):
        u = np.ones(64) * 3.0
        result = central_diffusion(u, dx=0.1, nu=0.01)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_heat_equation_decay(self):
        """Pure diffusion: u_t = nu * u_xx with u(x,0) = sin(x).
        Analytical: u(x,t) = exp(-nu*t) * sin(x).
        """
        nx = 256
        L = 2 * np.pi
        dx = L / nx
        x = np.linspace(0, L, nx, endpoint=False)
        nu = 0.1
        t_end = 0.5

        u = np.sin(x)

        def rhs(u):
            return central_diffusion(u, dx, nu)

        t = 0.0
        while t < t_end - 1e-14:
            dt = min(0.4 * dx**2 / nu, t_end - t)
            u = rk2_step(u, rhs, dt)
            t += dt

        u_exact = np.exp(-nu * t_end) * np.sin(x)
        error = np.max(np.abs(u - u_exact))
        assert error < 1e-3, f"Heat equation error too large: {error}"


class TestAdvection:
    def test_upwind_shape(self):
        u = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
        result = upwind_advection_flux(u, dx=2 * np.pi / 64)
        assert result.shape == u.shape

    def test_upwind_constant_is_zero(self):
        u = np.ones(64) * 2.0
        result = upwind_advection_flux(u, dx=0.1)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_lax_wendroff_shape(self):
        u = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
        result = lax_wendroff_advection_flux(u, dx=2 * np.pi / 64, dt=0.001)
        assert result.shape == u.shape


class TestCFL:
    def test_cfl_computation(self):
        u = np.array([1.0, -2.0, 3.0, -0.5])
        dx = 0.1
        dt = 0.01
        cfl = compute_cfl(u, dx, dt)
        assert cfl == pytest.approx(dt * 3.0 / dx)

    def test_stable_dt_respects_advection(self):
        u = np.array([10.0, -5.0, 3.0])
        dx = 0.1
        nu = 0.001
        dt = compute_stable_dt(u, dx, nu, cfl_target=0.5)
        assert dt <= 0.5 * dx / 10.0 + 1e-14

    def test_stable_dt_respects_diffusion(self):
        u = np.array([0.01])  # very small velocity
        dx = 0.1
        nu = 1.0  # large viscosity
        dt = compute_stable_dt(u, dx, nu, cfl_target=0.5, dt_max=1.0)
        assert dt <= 0.5 * dx**2 / nu + 1e-14


class TestTimeIntegrators:
    def test_euler_linear_ode(self):
        """du/dt = -u => u(t) = exp(-t)."""
        u = np.array([1.0])
        for _ in range(100):
            u = euler_step(u, lambda u: -u, dt=0.01)
        np.testing.assert_allclose(u, np.exp(-1.0), atol=0.02)

    def test_rk2_linear_ode(self):
        """du/dt = -u => u(t) = exp(-t). RK2 should be more accurate."""
        u = np.array([1.0])
        for _ in range(100):
            u = rk2_step(u, lambda u: -u, dt=0.01)
        np.testing.assert_allclose(u, np.exp(-1.0), atol=0.001)


class TestConvergenceOrder:
    """Verify spatial convergence rates using method of manufactured solutions."""

    def _run_burgers_diffusion_only(self, nx, nu, t_end):
        """Run pure diffusion (no advection) and return L2 error vs exact."""
        L = 2 * np.pi
        dx = L / nx
        x = np.linspace(0, L, nx, endpoint=False)
        u = np.sin(x)

        def rhs(u):
            return central_diffusion(u, dx, nu)

        t = 0.0
        while t < t_end - 1e-14:
            dt = min(0.4 * dx**2 / nu, t_end - t)
            u = rk2_step(u, rhs, dt)
            t += dt

        u_exact = np.exp(-nu * t_end) * np.sin(x)
        return np.sqrt(dx * np.sum((u - u_exact) ** 2))

    def test_diffusion_second_order(self):
        """Central diffusion should converge at second order."""
        nu = 0.1
        t_end = 0.1
        nx_list = [32, 64, 128]
        errors = [self._run_burgers_diffusion_only(nx, nu, t_end) for nx in nx_list]

        # Check convergence rate between successive refinements
        for i in range(len(errors) - 1):
            rate = np.log2(errors[i] / errors[i + 1])
            assert rate > 1.8, f"Expected ~2nd order, got rate={rate:.2f}"
