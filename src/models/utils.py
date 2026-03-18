"""Model I/O and parameter counting utilities.

Provides checkpoint save/load helpers that bundle the model weights with the training
configuration and best-validation metrics so that a checkpoint is fully self-describing.
The ``load_model`` function returns both the hydrated model and this metadata, making it
straightforward for downstream experiment scripts to reconstruct the correct
architecture from a checkpoint alone.
"""

import torch
import torch.nn as nn


def count_parameters(model):
    """Count the number of trainable (requires_grad) parameters in a model.

    Useful for verifying that the model stays lightweight — the default MLP has ~7 000
    parameters, keeping inference fast during multi-step rollout.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path, config=None, metrics=None):
    """Save model state_dict with optional training configuration and metrics metadata.

    The checkpoint dict always contains ``model_state_dict``; ``config`` and ``metrics``
    are included when provided so that downstream code can reconstruct the model
    architecture and inspect training quality without needing external files.

    Args:
        model: nn.Module to save.
        path: destination file path (conventionally .pt).
        config: optional training config dict (preserved in checkpoint for reproducibility).
        metrics: optional dict of training metrics (e.g., best_val_loss, epoch).
    """
    checkpoint = {"model_state_dict": model.state_dict()}
    if config is not None:
        checkpoint["config"] = config
    if metrics is not None:
        checkpoint["metrics"] = metrics
    torch.save(checkpoint, path)


def load_model(model, path, device="cpu"):
    """Load model weights from a checkpoint and return associated metadata.

    The caller must construct the model object first (with the correct architecture);
    this function populates its weights from the checkpoint. All non-weight entries
    (config, metrics, etc.) are returned as a metadata dict.

    Args:
        model: nn.Module instance (already constructed with matching architecture).
        path: checkpoint file path (.pt).
        device: target device for weight tensors ("cpu" or "cuda").

    Returns:
        Tuple of (model, metadata_dict) where metadata may contain "config" and
        "metrics" keys if they were saved with the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    return model, metadata
