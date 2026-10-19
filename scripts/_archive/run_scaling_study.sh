#!/bin/bash
source .venv/bin/activate
echo "Running tuning study for gamma=2"
python scripts/run_federated.py --config configs/attack_badnets.yaml --mode main --scale_factor 2.0 --rounds 50 --run_prefix "attack_scaling_gamma2_"

echo "Running tuning study for gamma=5"
python scripts/run_federated.py --config configs/attack_badnets.yaml --mode main --scale_factor 5.0 --rounds 50 --run_prefix "attack_scaling_gamma5_"

echo "Running tuning study for gamma=10"
python scripts/run_federated.py --config configs/attack_badnets.yaml --mode main --scale_factor 10.0 --rounds 50 --run_prefix "attack_scaling_gamma10_"
