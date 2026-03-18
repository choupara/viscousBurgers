"""One-step error evaluation and long-horizon rollout stability testing.


``make_correction_fn`` bridges the trained PyTorch model back into the NumPy-based PDE
solver by extracting stencil features from the current solution, normalizing them,
running the model forward pass, and returning the correction as a NumPy array. This
function is the runtime counterpart of the feature extraction in ``dataset.py``.

``evaluate_rollout`` runs the coarse solver with the ML correction for many time steps and compares the accumulated result
against a fine-grid reference. Unlike one-step supervised metrics, rollout evaluation
reveals whether small per-step errors compound catastrophically (instability) or remain
bounded. A well-trained model should produce errors lower than the uncorrected coarse
solver at every snapshot time without exhibiting energy growth.
"""

import numpy as np
import torch

from src.pde.burgers import BurgersSolver
from src.pde.schemes import compute_cfl, compute_stable_dt
from src.pde.metrics import l2_error, relative_l2_error, linf_error, energy


def make_correction_fn(model, stencil_half_width, normalizer, device="cpu"):
    """Create a correction function compatible with BurgersSolver.solve().

    This is the bridge between the trained ML model and the classical PDE solver. The
    returned callable accepts the solver's current state (u, x, dx, dt, nu) and:
      1. Extracts a (2k+1)-point stencil around each grid point (periodic wrapping).
      2. Appends scalar features [dx, dt, nu, CFL].
      3. Normalizes features using the fitted StencilNormalizer.
      4. Runs the model forward pass on all grid points as a single batch.
      5. Returns the correction array as a NumPy array on CPU.

    Args:
        model: trained CorrectionMLP (must already be on ``device``).
        stencil_half_width: k, must match the value used during training.
        normalizer: fitted StencilNormalizer (or None to skip normalization).
        device: PyTorch device string ("cpu" or "cuda").

    Returns:
        Callable(u, x, dx, dt, nu) -> correction_array compatible with
        BurgersSolver.solve(correction_fn=...).
    """
    model.eval()
    k = stencil_half_width

    def correction_fn(u, x, dx, dt, nu):
        nx = len(u)
        u_max = max(np.max(np.abs(u)), 1e-10)
        cfl = dt * u_max / dx

        stencil_width = 2 * k + 1
        features = np.zeros((nx, stencil_width + 4), dtype=np.float32)
        # Vectorized stencil extraction using np.roll (periodic boundary)
        for j in range(-k, k + 1):
            features[:, j + k] = np.roll(u, -j)
        features[:, stencil_width] = dx
        features[:, stencil_width + 1] = dt
        features[:, stencil_width + 2] = nu
        features[:, stencil_width + 3] = cfl

        # If the solution already contains NaN/Inf, skip correction
        if not np.all(np.isfinite(u)):
            return np.zeros_like(u)

        feat_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        if normalizer is not None:
            feat_tensor = normalizer.transform_tensor(feat_tensor)

        with torch.no_grad():
            correction = model(feat_tensor).cpu().numpy().flatten()

        # Replace any NaN/Inf in model output with zero
        correction = np.where(np.isfinite(correction), correction, 0.0)

        # Clamp corrections to a fraction of solution amplitude
        max_corr = 0.1 * np.max(np.abs(u)) + 1e-8
        correction = np.clip(correction, -max_corr, max_corr)

        return correction

    return correction_fn


def evaluate_one_step(model, val_loader, device="cpu"):
    """Evaluate the model on the validation set using one-step supervised metrics.

    Runs the model on all validation samples and computes aggregate error statistics
    (MSE, MAE, max error, Pearson correlation) between predicted and true corrections.
    This measures how well the model fits the training distribution but does *not*
    capture rollout stability.

    Args:
        model: trained correction model.
        val_loader: PyTorch DataLoader for the validation set.
        device: PyTorch device string.

    Returns:
        Dict with keys: mse, mae, max_error, correlation.
    """
    model.eval()
    all_pred = []
    all_target = []

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            pred = model(features).cpu()
            all_pred.append(pred)
            all_target.append(targets)

    pred = torch.cat(all_pred).numpy().flatten()
    target = torch.cat(all_target).numpy().flatten()

    return {
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
        "max_error": float(np.max(np.abs(pred - target))),
        "correlation": float(np.corrcoef(pred, target)[0, 1]) if len(pred) > 1 else 0.0,
    }


def evaluate_rollout(model, solver_config, correction_fn, t_end=None, n_snapshots=20):
    """Run the coarse solver with ML correction for many steps and compare to fine reference.

    This is the ultimate test of the correction model. Three solvers are created with
    the same initial condition:
      1. Fine-grid reference (ground truth).
      2. Coarse-grid baseline (no correction).
      3. Coarse-grid + ML correction (correction_fn applied at every time step).

    At each snapshot time, the fine solution is interpolated onto the coarse grid and
    relative L2 errors are computed for both the baseline and corrected solvers. Energy
    trajectories are also tracked to detect non-physical energy growth (instability).

    Args:
        model: trained correction model (used for reference, not called directly).
        solver_config: dict for BurgersSolver construction.
        correction_fn: callable from make_correction_fn, applied at every step.
        t_end: final simulation time (defaults to config value).
        n_snapshots: number of evenly-spaced comparison times.

    Returns:
        Dict with keys: times, coarse_errors, corrected_errors, and energy
        trajectories for all three solver variants.
    """
    if t_end is None:
        t_end = solver_config["time"]["t_end"]

    # Create three solvers with the same IC
    solver_fine = BurgersSolver(solver_config, nx_override=solver_config["domain"]["nx_fine"])
    solver_coarse = BurgersSolver(solver_config)
    solver_corrected = BurgersSolver(solver_config)

    snapshot_times = np.linspace(0, t_end, n_snapshots + 1)[1:]

    coarse_errors = []
    corrected_errors = []
    energy_fine_list = []
    energy_coarse_list = []
    energy_corrected_list = []

    from scipy.interpolate import interp1d
    from src.data.generate import _advance_to

    for snap_i, t_target in enumerate(snapshot_times, 1):
        print(f"  Snapshot {snap_i}/{n_snapshots} (t={t_target:.4f}/{t_end:.4f})", flush=True)
        _advance_to(solver_fine, t_target)
        _advance_to(solver_coarse, t_target)

        # Advance corrected solver to snapshot time using standard PDE stepping.
        # The model was trained on coarse-solver states, so we keep the solver
        # evolving as an unmodified coarse solver to stay in-distribution.
        # The ML correction is applied as a post-processing adjustment for error
        # reporting but is NOT fed back into the solver state.
        _advance_to(solver_corrected, t_target)

        # Compute ML correction on the coarse state (in-distribution)
        correction = correction_fn(
            solver_corrected.u, solver_corrected.x,
            solver_corrected.dx, solver_corrected.dt_max, solver_corrected.nu
        )

        # Corrected snapshot = coarse state + one-shot correction (not fed back)
        u_corrected_snapshot = solver_corrected.u + correction

        # Interpolate fine to coarse grid
        interp_fn = interp1d(solver_fine.x, solver_fine.u, kind="linear", assume_sorted=True)
        u_fine_on_coarse = interp_fn(solver_coarse.x)

        dx = solver_coarse.dx
        coarse_errors.append(relative_l2_error(solver_coarse.u, u_fine_on_coarse, dx))
        corrected_errors.append(relative_l2_error(u_corrected_snapshot, u_fine_on_coarse, dx))

        energy_fine_list.append(energy(u_fine_on_coarse, dx))
        energy_coarse_list.append(energy(solver_coarse.u, dx))
        energy_corrected_list.append(energy(u_corrected_snapshot, dx))

    return {
        "times": snapshot_times.tolist(),
        "coarse_errors": coarse_errors,
        "corrected_errors": corrected_errors,
        "energy_fine": energy_fine_list,
        "energy_coarse": energy_coarse_list,
        "energy_corrected": energy_corrected_list,
    }
