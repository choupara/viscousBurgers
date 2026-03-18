"""PyTorch dataset for supervised correction learning.

Bridges the raw ``.npz`` simulation data produced by ``generate.py`` with PyTorch's
DataLoader. Two modes are supported:

- **MLP mode** (default): Each sample is a single grid point. The feature vector
  concatenates a (2k+1)-point local stencil of the coarse solution centred on that
  point with four scalar simulation parameters [dx, dt, nu, CFL]. The target is the
  scalar correction Delta_u at that point. This flattens all (simulation, snapshot,
  grid point) triples into one large dataset — e.g. 150 sims * 20 snapshots *
  64 points = 192 000 samples.

- **CNN mode**: Each sample is one full snapshot. Input is the complete coarse field
  u_coarse of shape (1, nx) and the target is the full correction field Delta_u of
  shape (1, nx).

An optional ``StencilNormalizer`` transform can be attached to zero-mean, unit-variance
normalize the feature vectors at access time (MLP mode only).
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from src.pde.schemes import compute_cfl, compute_stable_dt


class CorrectionDataset(Dataset):
    """Dataset of (features, target) pairs for learning discretization corrections.

    In MLP mode each sample represents one grid point at one time snapshot: the input
    is a feature vector of length (2*k+1 + 4) containing the local stencil values and
    scalar simulation parameters, and the target is the scalar Delta_u correction. In
    CNN mode each sample is a full coarse solution field with the corresponding full
    correction field as the target.
    """

    def __init__(self, data_dir, stencil_half_width=3, mode="mlp", transform=None):
        """Load all .npz files from ``data_dir`` and extract supervised samples.

        Args:
            data_dir: directory containing .npz paired-simulation files from generate.py.
            stencil_half_width: half-width k of the local stencil (total width = 2k+1).
                Default 3 gives a 7-point stencil, yielding input_dim = 7 + 4 = 11.
            mode: "mlp" for per-point samples or "cnn" for full-field samples.
            transform: optional ``StencilNormalizer`` applied at access time (MLP only).
        """
        self.mode = mode
        self.stencil_half_width = stencil_half_width
        self.transform = transform

        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        if mode == "mlp":
            self._load_mlp_samples(npz_files)
        elif mode == "cnn":
            self._load_cnn_samples(npz_files)
        else:
            raise ValueError(f"Unknown mode '{mode}', choose 'mlp' or 'cnn'")

    def _load_mlp_samples(self, npz_files):
        """Extract per-point stencil features and scalar simulation parameters.

        For each grid point i in each snapshot of each simulation file, the feature
        vector is: [u[i-k], ..., u[i], ..., u[i+k], dx, dt, nu, CFL]. Periodic
        wrapping via modular indexing ensures correct stencil extraction at boundaries.
        The target is the scalar Delta_u[i] = u_fine_aligned[i] - u_coarse[i].
        """
        features_list = []
        targets_list = []
        k = self.stencil_half_width

        for fpath in npz_files:
            data = np.load(fpath, allow_pickle=True)
            x_coarse = data["x_coarse"]
            nx = len(x_coarse)
            dx = x_coarse[1] - x_coarse[0]
            nu = float(data["nu"][0])
            n_snapshots = int(data["n_snapshots"][0])

            for s in range(n_snapshots):
                u = data[f"u_coarse_{s}"]
                delta_u = data[f"delta_u_{s}"]
                t = float(data[f"t_{s}"][0])

                # Compute dt using the same logic as the solver at runtime
                u_max = max(np.max(np.abs(u)), 1e-10)
                dt = compute_stable_dt(u, dx, nu, cfl_target=0.5, dt_max=1e-3)
                cfl = compute_cfl(u, dx, dt)

                for i in range(nx):
                    # Extract stencil with periodic wrapping
                    stencil = np.array([u[(i + j) % nx] for j in range(-k, k + 1)])
                    scalar_feats = np.array([dx, dt, nu, cfl])
                    feat = np.concatenate([stencil, scalar_feats])
                    features_list.append(feat)
                    targets_list.append(delta_u[i])

        self.features = torch.tensor(np.array(features_list), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets_list), dtype=torch.float32).unsqueeze(-1)

    def _load_cnn_samples(self, npz_files):
        """Extract full-field samples for CNN mode.

        Each sample is one snapshot: the input is the full coarse solution field u_coarse
        of shape (1, nx) and the target is the full correction field Delta_u of shape
        (1, nx). The channel dimension is added for compatibility with Conv1d layers.
        """
        fields_list = []
        targets_list = []

        for fpath in npz_files:
            data = np.load(fpath, allow_pickle=True)
            n_snapshots = int(data["n_snapshots"][0])

            for s in range(n_snapshots):
                u = data[f"u_coarse_{s}"]
                delta_u = data[f"delta_u_{s}"]
                fields_list.append(u)
                targets_list.append(delta_u)

        self.fields = torch.tensor(np.array(fields_list), dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor(np.array(targets_list), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        if self.mode == "mlp":
            return len(self.features)
        return len(self.fields)

    def __getitem__(self, idx):
        if self.mode == "mlp":
            feat = self.features[idx]
            target = self.targets[idx]
            if self.transform is not None:
                feat = self.transform.transform_tensor(feat)
            return feat, target
        else:
            return self.fields[idx], self.targets[idx]
