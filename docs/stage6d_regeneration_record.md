# Stage 6D Regeneration Record

## 1. Stage 6D Objective
Analyze the impact of temporal attack injections (early, middle, late) vs matched counterfactual and no-attack baselines on federated unlearning efficacy.

## 2. Experimental Conditions
- **early**: Attack injected rounds 10-30.
- **middle**: Attack injected rounds 40-60.
- **late**: Attack injected rounds 70-90.
- **no_attack**: Clean baseline trajectory.
- **matched_counterfactual**: Clean trajectory but aligned closely to the initial state of the attacked runs.

## 3. Exact Timing Definitions
- Early (10-30)
- Middle (40-60)
- Late (70-90)

## 4. Client Sampling Schedule
Each run used the deterministic `client_sampling.json` preserved in `results/stage6d_summary/{condition}/attack/client_sampling.json` (or `no_attack`).

## 5. Attack Configuration
- Target client: 0
- Trigger size: 3 (canonical BadNets)
- Scaling Factor (gamma): 5.0
- Backdoor target class: 0

## 6. Random Seed
- Seed: 42 (Global deterministic seed used across Stage 6D).

## 7. Model/Configuration
- Architecture: SmallCNN
- Dataset: CIFAR-10 (Dirichlet partition alpha=0.5 or IID baseline depending on specific condition config).
- Configurations preserved in `results/stage6d_summary/*/config.yaml`.

## 8. Exact Scripts Used
- `scripts/run_stage6d_phase_a.py`
- `scripts/evaluate_phase_a.py`
- `scripts/run_stage6d_phase_b_full.py`
- `scripts/evaluate_phase_b.py`

## 9. Exact Canonical Source Artifacts Required
- `src/` module (SmallCNN, Data loaders)
- Stage 7C partitions (`results/stage7c_partitions/*`) if referenced.

## 10. Final Reported Metrics
Preserved in `results/stage6d_summary/*/metrics.csv`.

## 11. Final Figures/Analysis Outputs
Preserved in `results/stage6d_summary/*/figures/*.png`.

## 12. Location of Preserved Manifests
All compact metadata, metrics, and plots have been copied to `results/stage6d_summary/`.

## 13. Regeneration Instructions
1. Verify `src/` and `scripts/run_stage6d_phase_a.py` are present.
2. Ensure you have ~20 GB free disk space.
3. Run `python scripts/run_stage6d_phase_a.py` to regenerate the early/middle/late temporal attacks.
4. Run `python scripts/evaluate_phase_a.py` to regenerate metrics and figures.
5. (Optional) Run `phase_b` scripts for the unlearning components.
