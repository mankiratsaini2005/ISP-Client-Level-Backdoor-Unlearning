#!/bin/bash
set -e

echo "Running IID + Clean"
PYTHONPATH=. .venv/bin/python scripts/run_experiment_v2.py --config configs/v2_iid.yaml --output_dir results/iid_clean --mode clean

echo "Running IID + Attack"
PYTHONPATH=. .venv/bin/python scripts/run_experiment_v2.py --config configs/v2_iid.yaml --output_dir results/iid_attack --mode attack

echo "Running Non-IID + Clean"
PYTHONPATH=. .venv/bin/python scripts/run_experiment_v2.py --config configs/v2_noniid.yaml --output_dir results/noniid_clean --mode clean

echo "Running Non-IID + Attack"
PYTHONPATH=. .venv/bin/python scripts/run_experiment_v2.py --config configs/v2_noniid.yaml --output_dir results/noniid_attack --mode attack

echo "All 4 experiments finished."
