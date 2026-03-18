# ML-Corrected 1D Viscous Burgers Solver

A research-style prototype that integrates a classical numerical PDE solver with a machine-learning model that learns discretization-induced correction terms. The ML model learns to predict `Delta_u = u_fine - u_coarse`, correcting numerical errors from coarse-grid discretization.

## Project Goal

Build a research prototype that integrates a classical numerical PDE solver with a machine-learning model that learns discretization-induced correction terms. The goal is to demonstrate understanding of numerical methods, error sources, benchmarking, and controlled ML integration. The ML model corrects *numerical error*, not the physical solution itself.

## Setup

```bash
conda env create -f environment.yml
conda activate burgers-ml
```

## Quick Start

**Run the full pipeline:**
```bash
bash scripts/reproduce_all.sh
```

**Or step by step:**
```bash
# 1. Generate paired coarse/fine simulation data
bash scripts/gen_data.sh

# 2. Train the correction model
bash scripts/train.sh

# 3. Run baseline experiments
python -m src.experiments.run_baselines

# 4. Run with ML correction
python -m src.experiments.run_with_correction --model outputs/checkpoints/best_model.pt

# 5. Generate figures
python -m src.viz.make_figures
```

**Interactive demo:** Open `notebooks/quickstart_demo.ipynb`.

## Methodology

### Numerical Solver

The solver discretizes the 1D viscous Burgers equation on a periodic domain:

```
u_t + u * u_x = nu * u_xx
```

**Data Generation:**
- Coarse dt ≠ fine dt
- Coarse and fine advance independently but reach the same physical time *t*
- Coarse and fine use different spatial grids
- Fine solution is interpolated onto the coarse grid
- Store `u_coarse(t, x_coarse)` and `delta_u(t, x_coarse)`

**Training / Inference:**
- Given current coarse state `u(t, x_coarse)`
- Extract spatial stencil of size (2k+1) around each grid point
- Append 4 scalar parameters: `dx`, `dt`, `nu`, `CFL`
- Compute `dt` that the coarse solver would take **next**
- Predict correction `delta_u` consistent with **that** `dt`

$$\Delta u_i \approx \mathcal{F}(u_{i-k}, \dots, u_i, \dots, u_{i+k},\; \Delta x,\; \Delta t,\; \nu,\; \text{CFL})$$

- **Advection schemes**: First-order upwind (stable, diffusive) and second-order Lax-Wendroff (accurate, may oscillate near shocks). Both use periodic boundary conditions via `np.roll`.
- **Diffusion**: Standard 3-point central difference for `nu * u_xx`, subject to the von Neumann stability constraint `dt <= 0.5 * dx^2 / nu`.
- **Time integration**: Heun's method (RK2, second-order) as the default, with forward Euler as a fallback.
- **Adaptive time stepping**: CFL-based selection satisfying both advection (`dt <= CFL * dx / max|u|`) and diffusion stability limits.

### Data Generation

Paired simulations run the same physical problem on two grids:
- **Fine grid** (256 points): serves as the ground-truth reference.
- **Coarse grid** (64 points): has higher discretization error, especially near steep gradients and shocks.

At each snapshot time, the fine solution is interpolated onto the coarse grid and the correction target is computed:

```
Delta_u = u_fine_aligned - u_coarse
```

This Delta_u is what the ML model learns to predict. Three initial condition families test generalization: smooth sine sums, localized Gaussian bumps, and piecewise-constant Riemann problems.

### ML Correction Model

- **MLP** (primary): Pointwise prediction from a local (2k+1)-point stencil plus scalar features [dx, dt, nu, CFL]. Default: 7-point stencil, ~7,000 parameters with hidden layers [64, 64, 32]. Small by design — it runs at every grid point of every time step during rollout.
- **CNN** (alternative): 1D convolutional network predicting the full correction field from the coarse solution. Captures spatial correlations that the pointwise MLP treats independently.

Design choice: the final layer has no activation so corrections can be positive or negative. No batch normalization (small effective batch during rollout).

### Training

- **Optimizer**: Adam with learning rate 1e-3 and cosine annealing schedule.
- **Regularization**: Weight decay 1e-5, early stopping with patience 15 on validation loss.
- **Loss**: MSE between predicted and true corrections, with an optional smoothness penalty for the CNN model.
- **Normalization**: Per-feature zero-mean unit-variance scaling of inputs (critical because stencil values, dx, dt, nu, and CFL span very different scales).



### Evaluation

- **One-step metrics**: Standard supervised evaluation (MSE, MAE, correlation) on a held-out validation set.
- **Rollout testing**: The critical test — run the coarse solver with ML correction for many time steps and compare accumulated error vs the fine reference. Reveals whether per-step errors compound catastrophically (instability) or remain bounded.
- **Energy monitoring**: Track discrete kinetic energy over time. For the viscous Burgers equation energy must decrease monotonically; growth signals non-physical instability from the correction.

### Experiments

- **Baselines**: Coarse solver (upwind and Lax-Wendroff) vs fine reference across IC families and viscosity values.
- **Correction evaluation**: Coarse + ML correction vs baselines, with improvement ratios.
- **Ablations**: Sensitivity to stencil size, viscosity range, and IC family generalization.

## Project Structure

```
├── configs/                      # YAML configuration files
│   ├── burgers_default.yaml      # Solver defaults (grid, viscosity, scheme, CFL)
│   ├── dataset_gen.yaml          # Data generation (IC families, seeds, nu range)
│   └── train_correction.yaml     # Training (model architecture, optimizer, early stopping)
├── src/
│   ├── pde/                      # Numerical PDE solver components
│   │   ├── burgers.py            # BurgersSolver class with correction_fn hook
│   │   ├── schemes.py            # FD operators (upwind, Lax-Wendroff), diffusion, RK2, CFL
│   │   ├── initial_conditions.py # IC families (sine sums, Gaussians, Riemann problems)
│   │   └── metrics.py            # L2, relative L2, Linf errors and energy diagnostics
│   ├── data/                     # Data generation pipeline
│   │   ├── generate.py           # Paired coarse/fine simulations, correction targets
│   │   ├── dataset.py            # PyTorch Dataset (MLP per-point or CNN full-field)
│   │   └── transforms.py         # Feature normalization and target clipping
│   ├── models/                   # ML correction models
│   │   ├── correction_mlp.py     # Lightweight pointwise MLP (~7k params)
│   │   ├── correction_cnn.py     # 1D CNN for full-field correction
│   │   └── utils.py              # Checkpoint save/load, parameter counting
│   ├── train/                    # Training pipeline
│   │   ├── train_correction.py   # Config-driven training loop with early stopping
│   │   ├── losses.py             # MSE + optional smoothness penalty
│   │   └── eval.py               # One-step evaluation and multi-step rollout testing
│   ├── experiments/              # Experiment runners
│   │   ├── run_baselines.py      # Coarse vs fine reference benchmarks
│   │   ├── run_with_correction.py # ML-corrected solver evaluation
│   │   └── ablations.py          # Stencil size, viscosity, IC family sensitivity
│   └── viz/                      # Visualization
│       ├── plot_fields.py        # Solution profile comparisons
│       ├── plot_errors.py        # Error trajectory and energy drift plots
│       └── make_figures.py       # Batch figure regeneration from experiment results
├── scripts/                      # Shell automation
│   ├── gen_data.sh               # Data generation wrapper
│   ├── train.sh                  # Training wrapper
│   └── reproduce_all.sh          # Full pipeline: data -> train -> eval -> figures
├── notebooks/
│   └── quickstart_demo.ipynb     # Interactive walkthrough
└── tests/                        # pytest test suite (25 tests)
    ├── test_schemes.py           # FD operator and convergence order tests
    ├── test_burgers.py           # Solver integration and metric tests
    └── conftest.py               # Shared fixtures
```

## Reproducibility

The entire pipeline — data generation, training, evaluation, and figure generation — runs with a single command:

```bash
bash scripts/reproduce_all.sh
```

All results are saved to `outputs/` and can be regenerated from the config files alone.

## Testing

```bash
python -m pytest tests/ -v
```

25 tests covering:
- Convergence order verification (2nd order for central diffusion)
- Energy monotone decrease for viscous Burgers
- CFL stability constraint enforcement
- Solver integration (finite solution, scheme switching, correction hook)
- Error metric correctness
