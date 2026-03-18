"""1D convolutional neural network for full-field correction prediction.

An alternative to the pointwise MLP that processes the *entire* coarse solution field
at once, predicting a spatially-coherent correction field via a stack of 1D convolutions.
The convolutional receptive field naturally captures spatial correlations in the
discretization error that the MLP treats independently point-by-point.

Default architecture: Conv1d(1->16) -> ReLU -> Conv1d(16->32) -> ReLU ->
Conv1d(32->16) -> ReLU -> Conv1d(16->1). All convolutions use ``padding = kernel_size
// 2`` (same-padding) so the spatial dimension is preserved throughout.
"""

import torch
import torch.nn as nn


class CorrectionCNN(nn.Module):
    """1D CNN predicting the full correction field from the coarse solution.

    Uses same-padding to preserve the spatial dimension nx through all layers. The final
    layer has no activation so the correction can be positive or negative at each point.

    Input: (batch, 1, nx) — the full coarse solution field.
    Output: (batch, 1, nx) — the predicted correction field Delta_u.
    """

    def __init__(self, in_channels=1, hidden_channels=None, kernel_size=5, activation="relu"):
        """Build the CNN from the given architecture specification.

        Args:
            in_channels: number of input channels (1 for the coarse solution field).
            hidden_channels: list of channel widths for hidden conv layers (default
                [16, 32, 16], yielding ~5 300 parameters).
            kernel_size: spatial filter width (default 5). Must be odd for symmetric
                same-padding.
            activation: nonlinearity name — "relu", "tanh", or "gelu".
        """
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [16, 32, 16]

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        padding = kernel_size // 2

        layers = []
        prev_ch = in_channels
        for h_ch in hidden_channels:
            layers.append(nn.Conv1d(prev_ch, h_ch, kernel_size, padding=padding))
            layers.append(act_fn())
            prev_ch = h_ch
        layers.append(nn.Conv1d(prev_ch, 1, kernel_size, padding=padding))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass: (batch, 1, nx) -> (batch, 1, nx) correction field."""
        return self.net(x)
