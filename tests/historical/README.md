# Historical Test Suite

This directory preserves tests written for earlier exploratory prototype stages of the project. They are kept for complete archival provenance and auditability.

## Preserved Tests
1. `test_benchmark_freeze.py`:
   - Written for the September 9, 2026 exploratory 100-round prototype (`results/attack_scaling_final_gamma5_20260909_015903`, `results/unlearning_5b_kd_20260909_183734`).
   - These exploratory directories were superseded by the canonical 50-round Phase-1 and Phase-2 experiment benchmarks (`results/iid_attack`, `results/iid_clean`, `results/noniid_attack`, `results/noniid_clean`, `results_phase2/`).
2. `test_stage7b_counterfactual.py`:
   - Validates client exclusion in the historical Stage 7b runner (`scripts/_archive/run_federated.py` using `configs/_archive/federated_baseline.yaml`).

## Running Historical Tests
The active test suite (`pytest tests/`) excludes this directory by default via `pytest.ini`. To execute these historical tests explicitly:
```bash
PYTHONPATH=. pytest tests/historical/
```
