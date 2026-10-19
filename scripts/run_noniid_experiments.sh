#!/bin/bash
set -e
echo "Running Non-IID + Clean"
PYTHONPATH=. .venv/bin/python scripts/run_experiment_v2.py --config configs/v2_noniid.yaml --output_dir results/noniid_clean --mode clean

echo "Running Non-IID + Attack"
PYTHONPATH=. .venv/bin/python scripts/run_experiment_v2.py --config configs/v2_noniid.yaml --output_dir results/noniid_attack --mode attack

echo "Non-IID experiments finished."
