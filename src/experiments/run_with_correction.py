"""Run the coarse solver augmented with a trained ML correction and compare to baselines.

Loads a trained correction model checkpoint, reconstructs the model architecture from
the saved config, and runs multi-step rollout experiments across all IC families using
held-out seeds not seen during training. For each experiment, the script reports:
  - Final relative L2 error for the coarse-only solver (baseline).
  - Final relative L2 error for the coarse + ML correction solver.
  - Improvement ratio: (error_coarse - error_corrected) / error_coarse.

Positive improvement means the correction reduced the discretization error compared to
the uncorrected coarse solver. Results are saved as JSON for downstream plotting and
tabulation.

Usage:
    python -m src.experiments.run_with_correction --model outputs/checkpoints/best_model.pt
"""

import os
import argparse
import json

import numpy as np
import yaml
import torch

from src.pde.burgers import BurgersSolver
from src.pde.metrics import relative_l2_error, linf_error
from src.models.correction_mlp import CorrectionMLP
from src.models.correction_cnn import CorrectionCNN
from src.models.utils import load_model
from src.data.transforms import StencilNormalizer
from src.train.eval import make_correction_fn, evaluate_rollout


def run_with_correction(
    model_path,
    config_path="configs/burgers_default.yaml",
    output_dir="outputs/corrected",
):
    """Run the coarse solver with ML correction on held-out test cases.

    The model checkpoint is self-describing: it contains the training config, which is
    used to reconstruct the model architecture and determine the stencil width. The
    normalizer is loaded from the same directory as the checkpoint. Experiments use
    held-out IC seeds (10, 11, 12) that were not used in training data generation to
    test generalization.

    Args:
        model_path: path to trained model checkpoint (.pt file).
        config_path: path to solver configuration YAML.
        output_dir: directory where correction_results.json will be saved.

    Returns:
        List of result dicts, one per (ic_type, seed) combination, including error
        trajectories and improvement ratios.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    train_cfg = checkpoint.get("config", {})
    model_type = train_cfg.get("model", {}).get("type", "mlp")

    if model_type == "mlp":
        mlp_cfg = train_cfg["model"]["mlp"]
        model = CorrectionMLP(
            input_dim=mlp_cfg["input_dim"],
            hidden_dims=mlp_cfg["hidden_dims"],
            activation=mlp_cfg["activation"],
            dropout=0.0,
        )
    else:
        cnn_cfg = train_cfg["model"]["cnn"]
        model = CorrectionCNN(
            in_channels=cnn_cfg["in_channels"],
            hidden_channels=cnn_cfg["hidden_channels"],
            kernel_size=cnn_cfg["kernel_size"],
            activation=cnn_cfg["activation"],
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load normalizer
    norm_path = os.path.join(os.path.dirname(model_path), "normalizer.npz")
    normalizer = None
    if os.path.exists(norm_path) and model_type == "mlp":
        normalizer = StencilNormalizer()
        normalizer.load(norm_path)

    stencil_hw = train_cfg.get("stencil_half_width", 3)
    correction_fn = make_correction_fn(model, stencil_hw, normalizer, device)

    # Run experiments across IC families
    ic_families = ["sine_sum", "gaussian_bump", "riemann"]
    test_seeds = [10, 11, 12]  # held-out seeds not seen in training
    results = []
    total_experiments = len(ic_families) * len(test_seeds)
    exp_idx = 0

    for ic_type in ic_families:
        for seed in test_seeds:
            exp_idx += 1
            #print(f"\n[Experiment {exp_idx}/{total_experiments}] ic={ic_type}, seed={seed}", flush=True)
            test_cfg = dict(cfg)
            if ic_type == "sine_sum":
                ic_params = {"n_modes": 3, "amplitude_range": [0.5, 1.5], "seed": seed}
            elif ic_type == "gaussian_bump":
                ic_params = {"n_bumps": 1, "width": 0.1, "seed": seed}
            elif ic_type == "riemann":
                ic_params = {"u_left": 1.0, "u_right": 0.0}
            else:
                ic_params = {"seed": seed}
            test_cfg["initial_condition"] = {
                "type": ic_type,
                "params": ic_params,
            }

            rollout = evaluate_rollout(model, test_cfg, correction_fn)

            improvement = []
            for ce, cre in zip(rollout["coarse_errors"], rollout["corrected_errors"]):
                if ce > 1e-10:
                    improvement.append((ce - cre) / ce)
                else:
                    improvement.append(0.0)

            result = {
                "ic_type": ic_type,
                "seed": seed,
                "times": rollout["times"],
                "coarse_errors": rollout["coarse_errors"],
                "corrected_errors": rollout["corrected_errors"],
                "mean_improvement": float(np.mean(improvement)),
            }
            results.append(result)
            print(
                f"{ic_type:15s} seed={seed} | "
                f"coarse={rollout['coarse_errors'][-1]:.4f} | "
                f"corrected={rollout['corrected_errors'][-1]:.4f} | "
                f"improvement={np.mean(improvement)*100:.1f}%"
            )

    with open(os.path.join(output_dir, "correction_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/correction_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run corrected solver experiments")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/burgers_default.yaml")
    parser.add_argument("--output", default="outputs/corrected")
    args = parser.parse_args()
    run_with_correction(args.model, args.config, args.output)
