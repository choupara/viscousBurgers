"""Lightweight MLP for pointwise discretization-error correction.

The MLP is the primary correction model. It operates on a *single grid point* at a
time: given a local stencil of the coarse solution plus scalar simulation parameters,
it predicts the additive correction Delta_u that brings the coarse solution closer to
the fine-grid reference.

Design choices:
- Small network (~7 000 parameters with default [64, 64, 32] hidden layers) to keep
  inference cheap — the model is called at *every grid point* of *every time step*
  during a corrected rollout.
- No batch normalization, because during rollout the effective batch size is just nx
  (one grid at a time).
- Final layer has no activation so the correction can be positive or negative.
- Optional dropout for regularization during training (disabled during rollout via
  model.eval()).
"""

import torch
import torch.nn as nn


class CorrectionMLP(nn.Module):
    """Multi-layer perceptron predicting pointwise correction from local features.

    Default architecture (input_dim=11, hidden_dims=[64, 64, 32]):
        Input(11) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(1)

    The input vector of length 11 is: [stencil_values (7), dx, dt, nu, cfl], where the
    7-point stencil covers u[i-3] ... u[i+3] with periodic wrapping. The output is a
    scalar correction Delta_u for the centre grid point i.
    """

    def __init__(self, input_dim=11, hidden_dims=None, activation="relu", dropout=0.0):
        """Build the MLP from the given architecture specification.

        Args:
            input_dim: size of the input feature vector (stencil + scalars). Default 11
                corresponds to a 7-point stencil (k=3) plus 4 scalar features.
            hidden_dims: list of hidden layer widths (default [64, 64, 32]).
            activation: nonlinearity name — "relu", "tanh", or "gelu".
            dropout: dropout probability applied after each hidden layer (0 = disabled).
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 32]

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass: (batch, input_dim) -> (batch, 1) scalar corrections."""
        return self.net(x)
