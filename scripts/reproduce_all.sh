#!/bin/bash
set -e

echo "========================================="
echo "Full Reproducibility Pipeline"
echo "========================================="

echo ""
echo "Step 1: Generating data..."
bash scripts/gen_data.sh

echo ""
echo "Step 2: Training model..."
bash scripts/train.sh

echo ""
echo "Step 3: Running baselines..."
python -m src.experiments.run_baselines

echo ""
echo "Step 4: Running with correction..."
python -m src.experiments.run_with_correction --model outputs/checkpoints/best_model.pt

echo ""
echo "Step 5: Generating figures..."
python -m src.viz.make_figures

echo ""
echo "========================================="
echo "All done. Results in outputs/"
echo "========================================="
