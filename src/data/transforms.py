"""Normalization and clipping transforms for the correction training data.

Feature normalization is essential because the MLP input vector mixes quantities of very
different scales: stencil values (order ~1), grid spacing dx (order ~0.1), time step dt
(order ~1e-4), viscosity nu (order ~0.01), and CFL number (order ~0.5). Without
zero-mean unit-variance scaling the optimizer sees a badly conditioned loss landscape.

The ``TargetClipper`` is an optional transform that clips extreme correction values
(e.g., at shock locations) to prevent outliers from dominating the MSE loss during
training.
"""

import numpy as np
import torch


class StencilNormalizer:
    """Per-feature zero-mean unit-variance normalizer for MLP input vectors.

    Fitted once on the training set and then applied to both training and inference
    features, ensuring the model sees consistently scaled inputs. Supports both NumPy
    arrays (for batch fitting) and PyTorch tensors (for on-the-fly transform inside
    the Dataset). Parameters can be saved/loaded for use during rollout evaluation.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, features):
        """Fit mean and std from a numpy array of features.

        Args:
            features: (n_samples, n_features) numpy array
        """
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std[self.std < 1e-10] = 1.0  # avoid division by zero

    def transform(self, features):
        """Normalize features (numpy)."""
        return (features - self.mean) / self.std

    def transform_tensor(self, features):
        """Normalize a PyTorch tensor."""
        mean = torch.tensor(self.mean, dtype=features.dtype, device=features.device)
        std = torch.tensor(self.std, dtype=features.dtype, device=features.device)
        return (features - mean) / std

    def inverse_transform(self, features):
        """Undo normalization (numpy)."""
        return features * self.std + self.mean

    def save(self, path):
        """Save normalizer parameters."""
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        """Load normalizer parameters."""
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]


class TargetClipper:
    """Clip correction targets to suppress outliers at extreme gradient regions.

    Near shocks the correction Delta_u can be very large relative to the bulk of the
    data. Clipping to mean +/- n_sigma * std prevents these outliers from dominating
    the MSE loss, leading to a better model for the typical (non-shock) correction.
    """

    def __init__(self, n_sigma=5.0):
        self.n_sigma = n_sigma
        self.mean = None
        self.std = None

    def fit(self, targets):
        """Compute mean and std from targets.

        Args:
            targets: (n_samples,) numpy array
        """
        self.mean = targets.mean()
        self.std = targets.std()

    def transform(self, targets):
        """Clip targets to mean +/- n_sigma * std."""
        low = self.mean - self.n_sigma * self.std
        high = self.mean + self.n_sigma * self.std
        return np.clip(targets, low, high)
