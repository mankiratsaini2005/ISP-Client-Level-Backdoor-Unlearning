# Stage 7C D2 Attacked Run Recovery Report

## 1. Why the previous run was not resumable
The previous execution was interrupted externally at round 29. Because the training script used `--save_final_only`, no global checkpoint was saved for round 29 (only `best_global.pt` from round 28). More critically, the local optimizer momentum buffers and all Python/NumPy/PyTorch RNG states were not saved. Without these, it was mathematically impossible to exactly resume the DataLoader shuffling and optimizer behavior, meaning any continuation would geometrically diverge from the true federated trajectory required for FedEraser unlearning analysis.

## 2. New checkpointing safeguards
To harden the execution, `scripts/run_federated.py` was patched to explicitly save:
- `global_round_{round}.pt`: The actual global state at every round.
- `rng_state_round_{round}.pt`: The PyTorch, NumPy, and Python RNG states.
- `round_{round}_complete.json`: Explicit atomic completion markers to distinguish partially written rounds from fully successful ones.

## 3. D2 partition identity
The execution successfully loaded the exact D2 Client Indices from `results/stage7c_partitions/d2_indices.json`, preserving the target Label-Balanced / Size-Imbalanced condition (minimum client size 116, maximum size 12,266, balanced class proportions).

## 4. D2 attack configuration
The attack correctly enforced the Canonical A0 configuration:
- Target Client: 0
- Target Class: 0
- Trigger: BadNets (size 3)
- Poison Ratio: 0.10
- Gamma (Update Scaling): 5.0
- Attack Window: Rounds 30-100

## 5. BatchNorm/workability validation
A real DataLoader smoke test was run prior to execution. The smallest client (Client 11, with 116 samples) successfully produced batches of size 64 and 52. Since $116 \pmod{64} = 52 \neq 1$, no BatchNorm violations occurred.

## 6. Checkpoint & Client-Update Integrity
All 101 global checkpoints (`000` through `100`) and associated RNG states are present and pass integrity validation (no NaNs, no Infs). Client updates for all 100 rounds across all 10 selected clients per round were successfully captured. 

## 7. Attack Validity
- **Final Clean Accuracy**: 77.83%
- **Peak Clean Accuracy**: 78.25%
- **Final ASR**: 72.41%
- **Peak ASR**: 88.30%
The attack successfully injected the backdoor while preserving overall model utility. 

## 8. Readiness
The recovered D2 attacked trajectory is geometrically complete, mathematically continuous, and fully validated. It is ready for downstream Stage-7C unlearning and update-geometry decomposition.
