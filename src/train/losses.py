"""Loss functions for correction model training.

The primary loss is standard MSE between predicted and true corrections. For the CNN
model an optional smoothness penalty discourages spatially noisy correction fields that
could destabilize the solver during multi-step rollout. The penalty is the mean squared
difference of adjacent correction values, acting as a discrete total-variation-like
regularizer.
"""

import torch


def mse_loss(pred, target):
    """Standard mean squared error between predicted and true corrections.

    This is the primary training signal — minimizing the pointwise L2 distance between
    the model's predicted Delta_u and the ground-truth Delta_u from paired simulations.
    """
    return torch.mean((pred - target) ** 2)


def smoothness_penalty(pred):
    """Penalize large spatial gradients in the predicted correction field.

    Computes L_smooth = mean((Delta_u[i+1] - Delta_u[i])^2) over the spatial dimension.
    This encourages the CNN to produce smooth correction fields, reducing the risk of
    introducing high-frequency noise that can destabilize the solver during rollout.
    Only meaningful for CNN mode where pred has shape (batch, 1, nx).

    Args:
        pred: predicted correction field tensor of shape (batch, 1, nx).

    Returns:
        Scalar smoothness penalty (>= 0).
    """
    diff = pred[:, :, 1:] - pred[:, :, :-1]
    return torch.mean(diff**2)


def combined_loss(pred, target, smoothness_weight=0.01, mode="mlp"):
    """Weighted combination of MSE loss and optional smoothness penalty.

    In MLP mode (pointwise predictions) only MSE is used. In CNN mode the smoothness
    penalty is added with weight ``smoothness_weight`` to encourage spatially coherent
    corrections: L = MSE + smoothness_weight * L_smooth.

    Args:
        pred: model predictions — (batch, 1) for MLP, (batch, 1, nx) for CNN.
        target: ground truth corrections with matching shape.
        smoothness_weight: coefficient for the smoothness penalty term (CNN only).
        mode: "mlp" or "cnn".

    Returns:
        Scalar combined loss.
    """
    loss = mse_loss(pred, target)
    if mode == "cnn" and smoothness_weight > 0:
        loss = loss + smoothness_weight * smoothness_penalty(pred)
    return loss
