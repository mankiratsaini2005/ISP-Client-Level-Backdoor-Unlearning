# FedEraser Final Experiment Report

## 1. Objective
To execute the definitive 100-round FedEraser unlearning algorithm—aiming to reconstruct a clean global model without the malicious updates of Client 0—and explicitly evaluate its security behavior, utility, counterfactual fidelity, and resource costs.

## 2. Experimental Configuration
- **Model**: SmallCNN
- **Dataset**: CIFAR-10 (IID, 20 partitions, 2,500 samples/client)
- **Target**: Forget Client 0
- **Starting Point**: Round 0 canonical frozen model (`results/attack_scaling_final_gamma5_20260909_015903/checkpoints/global_round_000.pt`)
- **Calibration Settings**: E=2, B=64, LR=0.01, SGD (momentum 0.9, decay 5e-4)
- **Rounds**: 100
- **Evaluation Set**: 7,500 common frozen images.

## 3. FedEraser Procedure
At each of the 100 rounds:
1. Retained clients load their historical update $U$.
2. They perform 2 epochs of calibration training on their local data to generate $\hat{U}$.
3. The historical update is direction-calibrated to $\tilde{U}$ via $\tilde{U} = ||U||_2 \times \frac{\hat{U}}{||\hat{U}||_2}$.
4. The server aggregates the calibrated updates and advances the global state.

## 4. Approved Adaptations
**"FedEraser adapted to our 10-of-20 partial-participation PyTorch setting."**
1. **Partial-participation retained-client weight renormalization**: Weights are renormalized precisely over $S'_t$. Using the $N_k=2500$ natively within `aggregate_fedavg` yields `1/9` or `1/10` weights.
2. **PyTorch/BatchNorm/state-dict handling**: The magnitude normalization explicitly skips statistical float buffers (`running_mean`, `running_var`) to prevent statistical corruption, scaling only trainable float parameters.

## 5. Reconstruction Details
- 100 rounds reconstructed sequentially without NaNs or failures.
- In rounds where Client 0 participated, exactly 9 clients calibrated.
- Total Calibration Epochs: 1900.

## 6. Final Clean Accuracy
**75.00%** (Matches and slightly exceeds Stage 4's 74.48%)

## 7. Final ASR
**2.24%** (Eliminated the backdoor; strictly better than Stage 4's 3.02%)

## 8. Counterfactual Fidelity
**87.40%** agreement with the Stage 4 (Full Retraining) counterfactual model. 

## 9. Runtime
**Unlearning Time**: ~1680 seconds (28.0 minutes), excluding evaluation. 
*(Full Retraining took ~30 minutes).*

## 10. Communication
**Additional Communication**: 4.14 GB (4,144,204,000 bytes).
*(Original training was ~4.16 GB).*

## 11. Client Computation
**1900 total local calibration epochs.**
*(Full Retraining was 2000 epochs, but 1900 specifically removes the 100 Client 0 epochs).*

## 12. Storage
Only the final checkpoint and lightweight logs were retained (`results/unlearning_federaser_20260909_224318/`). No duplicate archives or round-by-round checkpoints were created.

## 13. Comparison Against Existing Baselines

| Method | Clean Accuracy | ASR | Counterfactual Fidelity | Time | Additional Communication | Client Work |
|---|---|---|---|---|---|---|
| Counterfactual Reference (Full Retraining) | 74.48% | 3.02% | Reference | ~30 min | ~4.16 GB | Full retraining |
| Historical Subtraction | 10.53% | 0.00% | Existing value | 4.85 s | 0 | None |
| Subtraction + KD | 56.00% | 3.09% | Existing value | ~15 s | 0 | None |
| **FedEraser adapted to our partial-participation PyTorch setting** | **75.00%** | **2.24%** | **87.40%** | **~28 min** | **~4.14 GB** | **1900 epochs** |

## 14. Interpretation
FedEraser demonstrates **excellent security behavior and utility preservation**. It achieves 75.00% clean accuracy (surpassing the 74.48% of Full Retraining) and drives the ASR down to 2.24%, completely removing the backdoor. Furthermore, it exhibits a high **counterfactual fidelity** (87.40% agreement), meaning it closely tracks the exact boundary decisions of true retraining. 

However, the **trade-offs** are steep:
- The computational cost (1900 epochs) and communication cost (4.14 GB) are virtually identical to Full Retraining from scratch.
- The unlearning time (~28 mins) is nearly identical to Full Retraining (~30 mins).

Thus, while mathematically sophisticated, FedEraser in a partial-participation non-convex setting acts essentially as a guided retraining scheme rather than a lightweight unlearning scheme.

## 15. Limitations
The primary limitation is extreme resource inefficiency. Unlearning requires nearly the same network footprint, client compute budget, and wall-clock time as throwing away the model and starting over.

## 16. Unexpected Behavior
None. The process ran extremely stably due to the BatchNorm exemptions.

## 17. Artifact Paths
- `results/unlearning_federaser_20260909_224318/`

## 18. Reproducibility Information
Seed: 42. Code executed sequentially on deterministic MPS hardware via `scripts/run_federaser.py`.

## 19. Confirmation of Canonical Preservation
Verified via `pytest tests/`. All canonical benchmark checkpoints, evaluation sets, and historical partitions remain 100% frozen and untouched.
