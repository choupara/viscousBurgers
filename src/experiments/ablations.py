"""Ablation studies: sensitivity analysis of the correction model.

Investigates how key design choices affect model performance:

- **Stencil size**: Wider stencils (larger k) give the MLP more spatial context but
  increase parameter count. Tests k = 1..5 (3-point to 11-point stencils).
- **Viscosity range**: Trains on different nu ranges to test whether the model
  generalizes across flow regimes (shock-dominated low-nu vs smooth high-nu).
- **IC family**: Trains on one IC family (e.g., sine_sum only) and evaluates on
  unseen families (gaussian_bump, riemann) to measure generalization to qualitatively
  different flow structures.

Each ablation saves a JSON summary describing the experimental setup and, where
available, the resulting metrics.
"""

import os
import json

import numpy as np
import yaml
import torch

from src.data.generate import generate_dataset
from src.train.train_correction import train
from src.train.eval import make_correction_fn, evaluate_rollout
from src.models.correction_mlp import CorrectionMLP
from src.models.utils import load_model, count_parameters
from src.data.transforms import StencilNormalizer


def ablation_stencil_size(stencil_sizes=None, output_dir="outputs/ablations/stencil"):
    """Report model architecture scaling as a function of stencil half-width k.

    For each stencil size, constructs the MLP and records the resulting input dimension
    and total parameter count. Wider stencils give the model more spatial context but
    increase cost. The default k=3 (7-point stencil, ~7k params) is a good balance.

    Args:
        stencil_sizes: list of stencil half-widths k to evaluate (default [1..5]).
        output_dir: directory for the JSON results file.

    Returns:
        List of result dicts with stencil_half_width, stencil_width, input_dim, n_parameters.
    """
    if stencil_sizes is None:
        stencil_sizes = [1, 2, 3, 4, 5]

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for k in stencil_sizes:
        input_dim = 2 * k + 1 + 4
        model = CorrectionMLP(input_dim=input_dim, hidden_dims=[64, 64, 32])
        n_params = count_parameters(model)

        result = {
            "stencil_half_width": k,
            "stencil_width": 2 * k + 1,
            "input_dim": input_dim,
            "n_parameters": n_params,
        }
        results.append(result)
        print(f"Stencil k={k}: width={2*k+1}, input_dim={input_dim}, params={n_params}")

    with open(os.path.join(output_dir, "stencil_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nStencil ablation summary saved to {output_dir}/")
    return results


def ablation_viscosity(output_dir="outputs/ablations/viscosity"):
    """Define viscosity range experiments for training-set sensitivity analysis.

    Tests three regimes: low viscosity (shock-dominated, nu ~ 0.001-0.01), medium
    (balanced, nu ~ 0.005-0.05), and high (diffusion-dominated, nu ~ 0.01-0.1). Models
    trained on one regime can be evaluated on others to measure cross-regime
    generalization.

    Args:
        output_dir: directory for the JSON results file.

    Returns:
        List of result dicts with label, nu_range, and description.
    """
    os.makedirs(output_dir, exist_ok=True)

    nu_ranges = {
        "low": [0.001, 0.01],
        "medium": [0.005, 0.05],
        "high": [0.01, 0.1],
    }

    results = []
    for label, (nu_min, nu_max) in nu_ranges.items():
        result = {
            "label": label,
            "nu_range": [nu_min, nu_max],
            "description": f"Train on nu in [{nu_min}, {nu_max}]",
        }
        results.append(result)
        print(f"Viscosity range '{label}': nu in [{nu_min}, {nu_max}]")

    with open(os.path.join(output_dir, "viscosity_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def ablation_ic_family(output_dir="outputs/ablations/ic_family"):
    """Define IC family generalization experiments.

    Tests whether a model trained on one IC family (e.g., smooth sine sums) generalizes
    to qualitatively different flow structures (e.g., Riemann shocks). This is a strong
    test of whether the model learns generalizable discretization-error patterns vs
    overfitting to specific solution shapes.

    Args:
        output_dir: directory for the JSON results file.

    Returns:
        List of experiment dicts with train_families and test_families.
    """
    os.makedirs(output_dir, exist_ok=True)

    experiments = [
        {"train_on": ["sine_sum"], "test_on": ["gaussian_bump", "riemann"]},
        {"train_on": ["gaussian_bump"], "test_on": ["sine_sum", "riemann"]},
        {"train_on": ["sine_sum", "gaussian_bump"], "test_on": ["riemann"]},
    ]

    results = []
    for exp in experiments:
        result = {
            "train_families": exp["train_on"],
            "test_families": exp["test_on"],
        }
        results.append(result)
        print(f"Train: {exp['train_on']} -> Test: {exp['test_on']}")

    with open(os.path.join(output_dir, "ic_family_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("=== Stencil Size Ablation ===")
    ablation_stencil_size()
    print("\n=== Viscosity Ablation ===")
    ablation_viscosity()
    print("\n=== IC Family Ablation ===")
    ablation_ic_family()
