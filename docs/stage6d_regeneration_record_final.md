# Stage 6D Final Preservation Record

## 1. Objective
Analyze the impact of temporal attack injections (early, middle, late) vs matched counterfactual and no-attack baselines on federated unlearning efficacy.

## 2. Exact Conditions & Timing Definitions
The exact temporal bounds as defined in `scripts/run_stage6d_phase_a.py`:
- **early**: Attack injected rounds 1-30.
- **middle**: Attack injected rounds 36-56.
- **late**: Attack injected rounds 64-100.
- **no_attack**: Clean baseline trajectory (no attack).
- **matched_counterfactual**: Clean trajectory, with Client 0 explicitly dropped post-sampling (`drop_clients_post_sampling: [0]`).

## 3. Exact Attack Configuration
Derived dynamically via modifier functions applied to `configs/attack_badnets.yaml`:
- Target client: 0
- Trigger size: 3 (canonical BadNets)
- Scaling Factor (gamma): 5.0
- Backdoor target class: 0

## 4. Sampling Schedule & Partition
- **Partition**: There is **no dependency** on Stage 7C partitions. Stage 6D dynamically generated a default partition (using `seed=42`) via `configs/attack_badnets.yaml`. 
- **Sampling Schedule**: There is **no dependency** on a canonical Stage 3B history. The client sampling was dynamically determined at runtime based on the fixed global `seed=42`.
- *(Note: The exact resulting `client_sampling.json` and `client_partitions.json` metadata for each run have been preserved in the summary directory as verification references).*

## 5. Scripts Required
- Phase A (Training): `scripts/run_stage6d_phase_a.py` and `scripts/evaluate_phase_a.py`
- Phase B (Unlearning): `scripts/run_stage6d_phase_b_full.py` and `scripts/evaluate_phase_b.py`
- Core Modules: `src/`, `scripts/run_federated.py`, `configs/attack_badnets.yaml`, and unlearning algorithms (`unlearn_stage5a.py`, `unlearn_stage5b.py`, `run_federaser.py`).

## 6. Canonical Dependencies
**None**. Stage 6D is completely self-contained. It generates its own source models, counterfactuals, and unlearning trajectories from the base codebase.

## 7. Preserved Artifacts (Compact Results)
Stored permanently in `results/stage6d_summary/`:
- Exact configuration YAMLs for each condition.
- Final `metrics.csv` for each condition.
- Metadata (`client_sampling.json`, `attack_metadata.json`, `client_partitions.json`).
- Final analysis figures (e.g., `test_acc_vs_round.png`, `attack_metrics_vs_round.png`).

## 8. Removed Raw Artifacts (Intentionally Deleted)
The raw PyTorch data (`results/archive_staging/stage_6d_temporal/`) was intentionally deleted to reclaim ~11.5 GB of SSD space. This included:
- Global model checkpoints (`global_round_XXX.pt`).
- Client update histories (`client_updates/round_XXX/client_XX.pt`).
- Unlearning artifacts generated during Phase B (checkpoints for FedEraser, KD, Subtraction).

## 9. Regeneration Procedure
Because the removed raw data consists purely of deterministic outputs derived from a fixed seed and unmodified codebase:
1. Ensure the `src/` codebase and `configs/attack_badnets.yaml` are exactly intact.
2. Ensure ~20 GB of free disk space is available.
3. Execute `python scripts/run_stage6d_phase_a.py`. This will exactly reproduce the early, middle, late, counterfactual, and no-attack `.pt` checkpoints.
4. Execute `python scripts/run_stage6d_phase_b_full.py` to regenerate the unlearned `.pt` checkpoints.
5. The resulting metric evaluations should perfectly match the preserved compact results in `results/stage6d_summary/`.

## 10. Reproducibility Caveats
Exact reproducibility is guaranteed *if and only if* the PyTorch pseudo-random number generator implementations and MPS backend determinism remain identical to the original execution environment. If hardware or PyTorch versions change, the exact bitwise trajectories may drift, but the scientific conclusions (the temporal trends) will remain valid.
