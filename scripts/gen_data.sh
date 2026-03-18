#!/bin/bash
set -e
echo "Generating paired coarse/fine simulation dataset..."
python -m src.data.generate --config configs/dataset_gen.yaml
echo "Done."
