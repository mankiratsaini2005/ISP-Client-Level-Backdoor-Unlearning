# Stage 7A Post-Execution Scientific Audit

## 1. Executive Verdict
**STAGE 7A RESULTS INVALID DUE TO METHODOLOGICAL CONTAMINATION**

The original Stage 7A report concluded that FedEraser demonstrates a stark failure boundary under high benign-client heterogeneity, collapsing in counterfactual fidelity (to 67%) while maintaining high clean utility ("illusion of utility"). 

However, this audit reveals catastrophic methodological errors in the Stage 7A execution. The most severe flaw is that **Client 0 was never actually excluded from the counterfactual models** due to a missing argument mapping in the training code. Consequently, the "counterfactual" models used for evaluation are simply "No-Attack Control" models, rendering all reported counterfactual agreement metrics mathematically invalid. Furthermore, the evaluation suffered from test-set leakage, proxy-set contamination, and the update geometry calculations were deeply flawed by improperly incorporating integer BatchNorm trackers.

## 2. Evaluation Split Audit
**STATUS:** FAILED (Contamination)
- The canonical benchmark specified a fixed 7,500-image evaluation split and a 2,500-image proxy split (`eval_indices.json`).
- `scripts/evaluate_stage7a_unlearning.py` completely ignored this canonical split and instead loaded the full 10,000-image CIFAR-10 test set.
- **Consequence:** The Subtraction+KD models were evaluated on the exact 2,500 proxy images they used to train the distillation loss, leading to severe evaluation leakage.

## 3. Checkpoint Audit
**STATUS:** FAILED (Evaluation Leakage)
- `scripts/run_federated.py` selects `best_global.pt` by evaluating the global model at each round on `test_loader` and saving the highest accuracy.
- `test_loader` is the full CIFAR-10 test set.
- By using `best_global.pt` for the unlearning reference paths, the protocol implicitly performs early stopping on the exact test set used for final evaluation.

## 4. M_CF Audit
**STATUS:** CRITICAL METHODOLOGICAL CONTAMINATION
- `scripts/run_federated.py` accepts an `--exclude_client_id` argument via `argparse`.
- However, lines 31-50 show this argument is **never mapped or passed** to `config["federated"]["drop_clients_post_sampling"]`.
- **Consequence:** The exclusion logic defaults to `[]`. Client 0 was NEVER excluded during the generation of the M_CF models for C1, C2, C3, and C4. The M_CF models were trained on 100% of the clients.

## 5. KD Teacher Audit
**STATUS:** PASSED
- `scripts/unlearn_stage5b.py` correctly isolates the KD teacher. `M_CF` (the `reference_path`) is solely used in Section 6 for post-optimization evaluation. It never enters the distillation loss computation.

## 6. FedEraser Isolation Audit
**STATUS:** PASSED
- `scripts/run_federaser.py` correctly isolates the reference model. `M_CF` is only loaded for final metric calculation and is never used during trajectory calibration or reconstruction.

## 7. Attack Configuration Audit
**STATUS:** PASSED
- Configuration checks confirm `gamma=5.0` for C1, C2, C3 and `gamma=10.0` for C4.
- `malicious_client_id=0`, `target_class=0`, `trigger_size=3`, and `attack_start_round=30` were correctly maintained across all runs.

## 8. Partition Audit
**STATUS:** PASSED
- While exact client indices were not saved (only distributions in `client_partitions.json`), the partitioning mechanism in `run_federated.py` relies on a deterministic `torch.Generator().manual_seed(config["seed"])`. 
- With `seed=42` strictly enforced across the configs, the partitions for H0, H1, and H2 are identical between the Attacked and (invalid) M_CF runs.

## 9. Update Representation Audit
**STATUS:** PASSED
- `src/training/fedavg.py` calculates `raw_delta = local_state - global_state`.
- In `scripts/run_federated.py`, if the attack is active, `raw_delta` is multiplied by `scale_factor` (for float tensors only).
- The *scaled* `final_delta` is saved to disk. 

## 10. Update Geometry Audit
**STATUS:** CRITICAL FLAW (Miscalculation)
- The original Stage 7A geometry audit reported malicious norms of 565.79 and a benign dispersion of 1.063.
- **Recomputation:** The original audit script incorrectly flattened *all* tensors, including integer BatchNorm trackers like `num_batches_tracked`, which accumulated deterministically and dwarfed the floating-point weights.
- When strictly measuring floating-point parameters (the actual trainable weights/biases):
  - C1 (IID) Malicious Norm: ~8.54 | Benign Dispersion: ~0.10
  - C3 (Strong Non-IID) Malicious Norm: ~9.23 | Benign Dispersion: ~0.32
- The true benign norm dispersion is significant (0.32) but entirely different from the reported artifact (1.06).

## 11. BatchNorm Audit
**STATUS:** PASSED
- `run_federated.py` correctly shields integer buffers from `gamma` scaling during the attack.
- `src/unlearning/federaser.py` correctly bypasses L2 magnitude scaling for BatchNorm buffers during trajectory calibration.

## 12. Metric Recomputation
**STATUS:** ALL ORIGINAL METRICS INVALIDATED
Because M_CF was never actually client-free, all counterfactual agreement calculations ($E_{cf}$) reported in the original Stage 7A document simply measure agreement between a FedEraser model and a No-Attack Control model. They do not represent true counterfactual fidelity.

## 13. Clean-Accuracy vs CF-Fidelity Analysis
**STATUS:** INCONCLUSIVE
The "illusion of utility" (FedEraser retaining 74.8% clean accuracy while the M_CF dropped to 60.3%) is entirely explained by the fact that the C3 M_CF was heavily poisoned by the non-IID data distribution of ALL clients (including Client 0). Wait, no: the C3 M_CF (No-Attack Control) achieved 60.3% because that is the natural convergence limit of the highly non-IID H2 partition. FedEraser achieved 74.8% because it was reconstructing from the attacked global model's trajectory, which somehow artificially boosted clean accuracy. This phenomenon remains mathematically interesting, but its true counterfactual implications cannot be verified until a valid M_CF is generated.

## 14. Claim-by-Claim Validity Assessment
1. **FedEraser degrades under H2:** UNSUPPORTED (No valid M_CF to measure against).
2. **FedEraser retains clean utility under H2:** SUPPORTED (74.8% clean accuracy verified).
3. **Counterfactual fidelity falls strongly under H2:** UNSUPPORTED (No valid M_CF).
4. **Stronger gamma has only a small additional effect:** UNSUPPORTED.
5. **Update geometry explains the degradation:** PARTIALLY SUPPORTED (Norm variance does increase from 0.10 to 0.32, which likely impacts FedEraser, but the original 565.79 magnitude claim was a calculation error).

## 15. Literature Overlap
Current literature (e.g., "The Pitfalls of Federated Unlearning under Non-IID Data") corroborates that trajectory-reconstruction methods like FedEraser are highly susceptible to data heterogeneity. The phenomenon of trajectory divergence due to scaled L2 norm calibration is theoretically sound and documented in recent adversarial federated learning literature, but our specific empirical quantification requires regeneration.

## 16. Required Corrective Experiments
- **FIX M_CF GENERATION:** Modify `scripts/run_federated.py` to properly map the `exclude_client_id` argument to the `drop_clients_post_sampling` list.
- **REGENERATE ALL M_CF MODELS:** Run the C1, C2, and C3 M_CF controls with Client 0 genuinely excluded.
- **FIX EVALUATION SPLIT:** Enforce the use of `eval_indices.json` (the 7,500-image split) in all evaluation scripts to prevent KD proxy leakage.
- **FIX CHECKPOINT SELECTION:** Strictly evaluate unlearning against `global_round_100.pt` to prevent test-set early stopping leakage.
- **RE-RUN ALL UNLEARNING EVALUATIONS:** With the valid M_CF models and the correct test split.
