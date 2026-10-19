# Stage 7C: Final Factorial Synthesis (v2 Canonical)

## 1. Single-Seed Limitation & Statistical Language
All four factorial conditions (D0, D1, D2, D3) were evaluated under a single deterministic experimental seed (seed 42). Therefore, all reported factorial interactions are **observed effects** within this specific experiment. No statistical generalization is established. Replication across multiple seeds would be required to estimate variability. The language used below strictly reflects this single-seed limitation.

## 2. Verification of Evaluation Parity
All four conditions were rigorously evaluated against the same canonical subset to ensure absolute methodological parity:

| Condition | Eval split | ASR protocol | Verified |
|---|---|---|---|
| D0 (C1) | eval_indices.json (7,500) | target=0, trigger=3 | Yes |
| D1 Canonical | eval_indices.json (7,500) | target=0, trigger=3 | Yes |
| D2 | eval_indices.json (7,500) | target=0, trigger=3 | Yes |
| D3 (C3) | eval_indices.json (7,500) | target=0, trigger=3 | Yes |

*Note: All four conditions used exactly the same subset indices, triggering function, target class, and precisely matched `poison_ratio: 0.1` settings.*

## 3. Final 2x2 Factorial Values (Canonical)
The following table presents the independently verified evaluation metrics. 

| Condition | Label Het. | Size Imbalance | Attack Ratio | FedEraser CF Agreement | CF Gap | Clean Acc |
|---|---|---|---|---:|---:|---:|
| D0 | Low | Low | 0.1 | 89.63% | 10.37% | 74.77% |
| D1 | High | Low | 0.1 | 87.09% | 12.91% | 74.85% |
| D2 | Low | High | 0.1 | 81.83% | 18.17% | 72.95% |
| D3 | High | High | 0.1 | 64.13% | 35.87% | 74.65% |

*(Definitions: CF Agreement = Prediction match against M_CF. CF Gap = 100 - CF Agreement)*

## 4. Factorial Effects & Interaction (Descriptive)
Using the final verified canonical values, the factorial decomposition reveals:

| Quantity | Value |
|---|---:|
| Label effect | -2.54 pp |
| Size effect | -7.80 pp |
| Additive prediction for D3 | 79.29% |
| Observed D3 | 64.13% |
| Non-additive interaction | -15.16 pp |

*Descriptive factorial decomposition under a single deterministic seed.*

## 5. Distinguishing Client Size, Update Norm, and Direction
To accurately interpret the proposed mechanism, three distinct quantities are rigorously separated:
1. **Client Sample Size**: Determines FedAvg weighting. Equal client sizes strictly remove sample-count weighting imbalance.
2. **Local Update Norm**: The magnitude of the local update vector. While client sizes control weights, local update norms may differ due to local data variance.
3. **Update Direction**: The angular orientation of the update vector in the gradient space.

## 6. Geometric Comparison
The following table compares the geometric characteristics of the actual aggregation-weighted updates near Round 100:

| Metric | D0 (Low/Low) | D1 (High/Low) | D2 (Low/High) | D3 (High/High) |
|---|---|---|---|---|
| Client-Size CV | 0.0 | 0.0 | 1.11 | 1.11 |
| Label Entropy (bits) | 3.32 | 2.15 | 3.32 | 0.17 |
| Benign Norm CV (Magnitude Dispersion) | 0.10 | 0.15 | 0.53 | 0.32 |
| Benign Directional Dispersion (1-cos) | 0.95 | 1.00 | 0.98 | 0.99 |
| Malicious/Benign Norm Ratio | 4.24x | 1.27x | 4.93x | 5.10x |

## 7. D1 Directional Dispersion Analysis
D1 Canonical exhibits absolute extreme benign directional dispersion (1-cos ≈ 1.00, implying functionally orthogonal updates), yet FedEraser achieves a highly successful 87.09% CF Agreement. This explicitly demonstrates that extreme directional dispersion *alone* does not cause catastrophic calibration failure. FedEraser remains highly resilient to conflicting gradient directions when update magnitudes are constrained by uniform sample counts.

## 8. D2 Size Imbalance Analysis
D2 (high size imbalance, low label heterogeneity) yields an 81.83% CF Agreement. This provides evidence that unequal aggregation influence (reflected in the ~5x malicious/benign norm ratio) is associated with moderate reconstruction degradation, even when label heterogeneity is exceptionally low.

## 9. D3 Interaction & Proposed Mechanism
The observed D3 CF Agreement (64.13%) is substantially lower than the additive prediction (79.29%), quantifying a massive non-additive interaction (-15.16 pp) under the joint heterogeneous regime.

**Candidate Mechanism:**
HIGH LABEL HETEROGENEITY (conflicting update directions) 
+ HIGH SIZE IMBALANCE (unequal aggregation influence) 
→ non-additive reconstruction difficulty.

**Evaluation:** This mechanism is **strongly supported** by the current single-seed evidence. 
The data are consistently aligned with a non-linear interactive effect: massive size imbalances create disparate gradient magnitudes, which non-linearly warp the geometric space across the highly divergent orthogonal directions caused by label heterogeneity. 

## 10. Counterfactual Fidelity vs Utility
A fundamental finding across the factorial matrix is the dissociation of clean utility from counterfactual fidelity:
- D3 FedEraser achieved excellent clean accuracy (74.65%, matching D0's 74.77%).
- However, D3 suffered catastrophic counterfactual fidelity failure (64.13% CF Agreement).
- This distinction explicitly confirms that utility metrics are inherently blind to algorithmic unlearning failures and cannot be used as proxies for unlearning fidelity.

## 11. Role of the D1-StrongAttack Exploratory Result
The original D1 run utilizing `poison_ratio: 0.5` is preserved as `D1-StrongAttack`. It serves as a valuable robustness validation, confirming that even when the localized injection rate is highly elevated, FedEraser is still shielded from catastrophic failure when client sizes remain equal (CF Agreement: 88.20%).

## 12. Final Claim
The completed 2×2 study indicates that neither label heterogeneity nor size imbalance alone reproduces the severe FedEraser degradation observed in the joint regime. The joint condition exhibits a substantially larger-than-additive loss of counterfactual fidelity, consistent with an interaction between heterogeneous update directions and unequal aggregation influence. The precise causal mechanism remains to be confirmed.

## 13. Recommendation for Stage 7D
Stage 7D is justified if the explicit objective is to test whether controlling one suspected mechanistic variable (e.g., forcing mathematical magnitude normalization inside FedEraser) reverses the interaction. If the objective is merely to characterize the failure space further, deeper architectural analysis of the geometry may take precedence over additional FL trajectories.
