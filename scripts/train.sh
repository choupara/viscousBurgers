#!/bin/bash
set -e
echo "Training correction model..."
python -m src.train.train_correction --config configs/train_correction.yaml
echo "Done."
