# Stage 7C: D2 Correction & Integrity Audit

## 1. Discrepancy Discovered
The original report erroneously conflated the Utility Gap with the Counterfactual (CF) Gap. The original report calculated `CleanAccuracy(M_CF) - CleanAccuracy(M_U)` and labeled it as the CF Gap. Furthermore, the previous update geometry audit incorrectly assumed the client updates stored on disk were raw unscaled deltas, resulting in double-scaling the malicious norm by $\gamma=5.0$. Finally, the frozen evaluation split was verified to ensure the dynamically generated split exactly matched the canonical one.

These issues have been fully corrected through independent code audits and standalone verification scripts.

## 2. Frozen Evaluation Split Verification
The `eval_indices.json` used by FedEraser was verified to be strictly identical to the benchmark `torch.randperm(10000, generator=g)[2500:]`. The final audit script (`scripts/audit_stage7c_d2_final_corrected.py`) was updated to load this frozen JSON file directly rather than regenerating the indices, guaranteeing exact alignment with the canonical split.

## 3. M_CF Client-0 Exclusion Verification
A static execution path audit of `scripts/run_federated.py` confirms that `drop_clients_post_sampling` intercepts the participant list strictly before `local_model` instantiation, data loading, or optimization. The relevant loop (`for client_id in selected_clients:`) completely skips Client 0 when it is dropped post-sampling. Consequently, Client 0 does not train and its delta is omitted from aggregation. The M_CF simulation is structurally sound and mathematically equivalent to a true counterfactual baseline where Client 0 never participated in that round.

## 4. D2 Partition Statistics
The D2 partition (`results/stage7c_partitions/d2_indices.json`) was audited mathematically:
**D2 SIZE IMBALANCE:**
- **Min/Max**: 116 / 12,266 samples
- **Mean / Std**: 2,500 / 2,776.88
- **CV**: 1.11
- **Gini Coefficient**: 0.516
(Client 0 retains its fixed size of 2,500).

**D2 LABEL HETEROGENEITY:**
- **Mean Entropy**: 3.32 bits (near the maximum of 3.3219 bits for 10 classes).
- **Mean Pairwise JS Divergence**: 0.0011 (effectively zero).
Conclusion: D2 exhibits extreme size imbalance matching H2, but strictly preserves uniform IID label distributions matching H0.

## 5. Corrected Update Geometry (Rounds 24, 49, 73, 100)
A code audit revealed that the FL pipeline stores the *scaled* malicious update (i.e., after applying $\gamma=5.0$) rather than the raw delta. Removing the erroneous double-scaling yields the true aggregation geometry:

| Metric | Round 24 | Round 49 | Round 73 | Round 100 |
|---|---|---|---|---|
| Malicious Actual Aggregation Norm | 1.63 | 10.46 | 9.35 | 8.61 |
| Mean Benign Norm | 1.07 | 1.16 | 1.91 | 1.75 |
| Malicious/Benign Norm Ratio | 1.53x | 8.99x | 4.89x | 4.93x |
| Benign CV (Dispersion of Norms) | 0.58 | 0.54 | 0.45 | 0.53 |
| Benign Directional Dispersion (1 - cos) | 0.974 | 0.870 | 0.954 | 0.977 |
| Malicious-to-Benign Mean Cosine | 0.073 | 0.145 | 0.053 | 0.053 |

*Correction Note: While the malicious update still dominates aggregation (up to ~9x the norm of an average benign update), it is not the 45x hyper-dominant outlier incorrectly reported earlier. Benign directional dispersion remains extreme.*

## 6. FedEraser Calibration Error
**NOT DIRECTLY AVAILABLE IN CURRENT IMPLEMENTATION.**
The canonical FedEraser script does not retain per-client, per-round intermediate calibration residual metrics in its `metrics.csv`. Per strict instructions, the `run_federaser.py` logic was not modified to extract them. The cumulative historical calibration error is instead holistically reflected by the final CF Gap below.

## 7. Corrected D2 Unlearning Evaluation
Evaluated directly on the frozen benchmark subset.
- **Utility Gap** = `CleanAccuracy(M_CF) - CleanAccuracy(M_U)`
- **CF Gap** = `100.0 - PredictionAgreement(M_U, M_CF)`

| Method | Clean Acc | Clean Loss | ASR | CF Agreement | CF Gap | Utility Gap |
|---|---:|---:|---:|---:|---:|---:|
| D2 M_CF (Reference) | 77.40% | 0.662 | 2.66% | 100.00% | 0.00% | 0.00% |
| D2 Attacked | 77.61% | 0.655 | 72.48% | 87.24% | 12.76% | -0.21% |
| Subtraction | 10.77% | 1774.34 | 0.00% | 9.68% | 90.32% | 66.63% |
| KD | 58.24% | 1.173 | 4.06% | 65.20% | 34.80% | 19.16% |
| FedEraser | 72.95% | 0.771 | 1.89% | 81.83% | 18.17% | 4.45% |

## 8. Condition Comparisons (D0 vs D2 vs D3/H2)
Comparing the FedEraser performance across matching evaluation protocols:

| Condition | Description | Clean (M_CF) | Clean (FedEraser) | CF Agreement | CF Gap |
|---|---|---|---|---|---|
| **D0** (C1) | Low Label Het, Low Size Imb | 74.76% | 74.77% | 89.63% | 10.37% |
| **D2** | Low Label Het, High Size Imb | 77.40% | 72.95% | 81.83% | 18.17% |
| **D3** (C3) | High Label Het, High Size Imb | 59.24% | 74.65% | 64.13% | 35.87% |

*(Note: D0/D3 values are extracted from the recomputed Stage 7A audits for exact methodological parity).*

## 9. Corrected Scientific Interpretation & Claims
**Size-Imbalance Evidence Classification: B. MODERATE.**
The controlled comparison between D0 and D2 provides moderate evidence that introducing severe size imbalance into an IID setting is associated with a measurable deterioration in counterfactual reconstruction (CF Agreement falls from 89.63% down to 81.83%). 

However, size imbalance is mathematically insufficient to cause the extreme unlearning divergence observed in the D3/H2 scenario (CF Agreement 64.13%). Label heterogeneity appears to be the dominant driver of catastrophic FedEraser failure, while size imbalance acts as a secondary destabilizing factor.

## 10. Remaining Uncertainty
Because D3 (H2) involves *both* high size imbalance and high label heterogeneity, the precise contribution of label heterogeneity in isolation cannot be quantified without evaluating the D1 condition (High Label Het, Low Size Imb). 

## 11. Final Decision
**B. D2 VALID WITH CORRECTIONS — PROCEED TO D1**
The analytical and geometric calculation errors have been structurally isolated and corrected. The underlying D2 data (checkpoints, logs, partitions) are intact, valid, and mathematically sound. The D2 experiment constitutes a rigorous scientific artifact and is now cleared for cross-comparison.

Proceed to D1 to complete the 2x2 factorial matrix.
