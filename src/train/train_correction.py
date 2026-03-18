"""End-to-end config-driven training loop for the correction model.

Orchestrates the full training pipeline:
  1. Load the dataset of (features, Delta_u) samples from generated .npz files.
  2. Fit a StencilNormalizer on the training features (MLP mode) and save it for use
     during rollout evaluation.
  3. Split into training and validation sets (configurable fraction).
  4. Instantiate the model (MLP or CNN) from the YAML config.
  5. Train with Adam optimizer and cosine-annealing learning rate schedule.
  6. Apply early stopping on validation loss (patience configurable).
  7. Save the best checkpoint (weights + config + metrics) for downstream use.

All hyperparameters are read from ``configs/train_correction.yaml`` — there are no
hard-coded training settings outside of defaults for missing keys.

Usage:
    python -m src.train.train_correction --config configs/train_correction.yaml
"""

import os
import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from src.data.dataset import CorrectionDataset
from src.data.transforms import StencilNormalizer
from src.models.correction_mlp import CorrectionMLP
from src.models.correction_cnn import CorrectionCNN
from src.models.utils import count_parameters, save_model
from src.train.losses import combined_loss


def train(config_path):
    """Main training entry point. Loads config and runs the full pipeline.

    Steps: load data -> fit normalizer -> train/val split -> build model -> train loop
    with early stopping -> save best checkpoint.

    Args:
        config_path: path to train_correction.yaml.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    model_cfg = cfg["model"]
    mode = model_cfg["type"]
    data_dir = cfg.get("data_dir", "data/generated")
    stencil_hw = cfg.get("stencil_half_width", 3)

    dataset = CorrectionDataset(data_dir, stencil_half_width=stencil_hw, mode=mode)
    print(f"Dataset size: {len(dataset)} samples")

    # Fit normalizer (MLP mode only)
    normalizer = None
    if mode == "mlp":
        normalizer = StencilNormalizer()
        all_features = dataset.features.numpy()
        normalizer.fit(all_features)
        dataset.transform = normalizer

        norm_dir = cfg["io"]["checkpoint_dir"]
        os.makedirs(norm_dir, exist_ok=True)
        normalizer.save(os.path.join(norm_dir, "normalizer.npz"))
        print("Normalizer fitted and saved")

    # Train/val split
    train_cfg = cfg["training"]
    val_fraction = train_cfg["val_fraction"]
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"])

    # Create model
    if mode == "mlp":
        mlp_cfg = model_cfg["mlp"]
        model = CorrectionMLP(
            input_dim=mlp_cfg["input_dim"],
            hidden_dims=mlp_cfg["hidden_dims"],
            activation=mlp_cfg["activation"],
            dropout=mlp_cfg["dropout"],
        )
    else:
        cnn_cfg = model_cfg["cnn"]
        model = CorrectionCNN(
            in_channels=cnn_cfg["in_channels"],
            hidden_channels=cnn_cfg["hidden_channels"],
            kernel_size=cnn_cfg["kernel_size"],
            activation=cnn_cfg["activation"],
        )

    model = model.to(device)
    print(f"Model: {mode.upper()}, {count_parameters(model)} trainable parameters")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    loss_cfg = cfg["loss"]
    smoothness_weight = loss_cfg.get("smoothness_weight", 0.01)

    # Training loop
    os.makedirs(cfg["io"]["checkpoint_dir"], exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    patience = train_cfg["patience"]

    for epoch in range(train_cfg["epochs"]):
        # Train
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(features)
            loss = combined_loss(pred, targets, smoothness_weight, mode)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                pred = model(features)
                loss = combined_loss(pred, targets, smoothness_weight, mode)
                val_loss_sum += loss.item()
                n_val_batches += 1

        val_loss = val_loss_sum / max(n_val_batches, 1)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | lr: {lr:.2e}")

        # Early stopping / checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(
                model,
                os.path.join(cfg["io"]["checkpoint_dir"], "best_model.pt"),
                config=cfg,
                metrics={"best_val_loss": best_val_loss, "epoch": epoch + 1},
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Best model saved to {cfg['io']['checkpoint_dir']}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train correction model")
    parser.add_argument("--config", default="configs/train_correction.yaml")
    args = parser.parse_args()
    train(args.config)
