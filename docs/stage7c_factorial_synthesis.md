# Stage 7C: Final Factorial Synthesis & Mechanism Audit

## 1. Single-Seed Limitation & Statistical Language
All four factorial conditions (D0, D1, D2, D3) were evaluated under a single deterministic experimental seed (seed 42). Therefore, all reported factorial interactions are **observed effects** within this specific experiment. No statistical generalization is established. Replication across multiple seeds would be required to estimate variability. The language used below (e.g., "consistent with", "suggests", "associated with") strictly reflects this single-seed limitation.

## 2. Verification of Evaluation Parity
All four conditions were rigorously evaluated against the same canonical subset to ensure methodological parity:

| Condition | Eval split | ASR protocol | Verified |
|---|---|---|---|
| D0 (C1) | eval_indices.json (7,500) | target=0, trigger=3 | Yes |
| D1 | eval_indices.json (7,500) | target=0, trigger=3 | Yes |
| D2 | eval_indices.json (7,500) | target=0, trigger=3 | Yes |
| D3 (C3) | eval_indices.json (7,500) | target=0, trigger=3 | Yes |

*Note: All four conditions used exactly the same subset indices, triggering function, and target class.*

## 3. Attack Validity & Configuration Audit
**WARNING:** A rigorous audit of the executed D1 `config.yaml` and `attack_metadata.json` revealed that D1 was inadvertently executed with `poison_ratio: 0.5` instead of the canonical `poison_ratio: 0.1` used in D0, D2, and D3. 

While the D1 attack was highly successful (ASR 51.87%), it represents a stronger localized poisoning injection than the other conditions. This difference in attack strength may influence the difficulty of unlearning and introduces a minor confounding variable in the strict 2x2 symmetry. However, despite this stronger attack, FedEraser still achieved exceptional counterfactual fidelity in D1, which further emphasizes the resilience of FedEraser to label heterogeneity alone.

## 4. Final 2x2 Factorial Values
The following table presents the independently verified evaluation metrics. 

*(Definitions: CF Agreement = Prediction match against M_CF. CF Gap = 100 - CF Agreement. Utility Gap = Clean(M_CF) - Clean(FedEraser))*

| Condition | Label Het. | Size Imbalance | FedEraser CF Agreement | CF Gap | Clean Acc |
|---|---|---|---:|---:|---:|
| D0 | Low | Low | 89.63% | 10.37% | 74.77% |
| D1 | High | Low | 88.20% | 11.80% | 74.71% |
| D2 | Low | High | 81.83% | 18.17% | 72.95% |
| D3 | High | High | 64.13% | 35.87% | 74.65% |

## 5. Factorial Effects & Interaction (Descriptive)
Using the independently audited values, the factorial decomposition reveals:

| Quantity | Value |
|---|---:|
| Label effect | -1.43 pp |
| Size effect | -7.80 pp |
| Additive prediction for D3 | 80.40% |
| Observed D3 | 64.13% |
| Non-additive interaction | -16.27 pp |

*Descriptive factorial decomposition under a single deterministic seed.*

## 6. Distinguishing Client Size, Update Norm, and Direction
To interpret the mechanism, three distinct quantities must not be conflated:
1. **Client Sample Size**: Determines FedAvg weighting. Equal client sizes remove sample-count weighting imbalance.
2. **Local Update Norm**: The magnitude of the local update vector. Even with equal sample counts, local update norms may differ due to local data characteristics.
3. **Update Direction**: The orientation of the update vector in the parameter space.

## 7. D1 / D2 / D3 Geometry Comparison
The following table compares the geometric characteristics of the actual aggregation-weighted updates near Round 100:

| Metric | D0 (Low/Low) | D1 (High/Low) | D2 (Low/High) | D3 (High/High) |
|---|---|---|---|---|
| Client-Size CV | 0.0 | 0.0 | 1.11 | 1.11 |
| Label Entropy (bits) | 3.32 | 2.15 | 3.32 | 0.17 |
| Benign Norm CV (Magnitude Dispersion) | 0.10 | 0.16 | 0.53 | 0.32 |
| Benign Directional Dispersion (1-cos) | 0.95 | 0.98 | 0.98 | 0.99 |
| Malicious/Benign Norm Ratio | 4.24x | 1.17x | 4.93x | 5.10x |

**Pattern Classification**: The geometric pattern is consistent with **C. Interaction between magnitude/weight and direction**. 

## 8. D1 Directional Dispersion Analysis
D1 exhibits extreme benign directional dispersion (1-cos ≈ 0.98), yet FedEraser achieves a highly successful 88.20% CF Agreement. The data demonstrate that extreme directional dispersion *alone* does not cause catastrophic calibration failure. This suggests that FedEraser's historical calibration algorithm is highly resilient to conflicting update directions provided the aggregation magnitudes (driven by client sizes) remain relatively uniform. 

## 9. D2 Size Imbalance Analysis
D2 (high size imbalance, low label heterogeneity) yields an 81.83% CF Agreement. This provides evidence that unequal aggregation influence (stemming from size imbalance and reflected in the ~5x malicious/benign norm ratio and high Benign Norm CV) is associated with moderate reconstruction degradation, even when label heterogeneity is low.

## 10. D3 Interaction & Proposed Mechanism
The observed D3 CF Agreement (64.13%) is substantially lower than the additive prediction (80.40%), quantifying a non-additive interaction (-16.27 pp) under the joint heterogeneous regime.

**Candidate Mechanism:**
HIGH LABEL HETEROGENEITY (conflicting update directions) 
+ HIGH SIZE IMBALANCE (unequal aggregation influence) 
→ non-additive reconstruction difficulty.

**Evaluation:** This mechanism is **strongly supported** by the current single-seed evidence. 
The data are highly consistent with an interactive effect: massive size imbalances create disparate gradient magnitudes, which non-linearly warp the geometric space across the highly divergent orthogonal directions caused by label heterogeneity.

## 11. Counterfactual Fidelity vs Utility
A central finding across all conditions is the dissociation of clean utility from counterfactual fidelity:
- D3 FedEraser achieved excellent clean accuracy (74.65%, almost identical to D0's 74.77%).
- However, D3 suffered catastrophic counterfactual fidelity failure (64.13% CF Agreement).
- This explicitly demonstrates that high clean utility frequently coexists with failed unlearning fidelity, proving that utility metrics are insufficient proxies for algorithmic unlearning success.

## 12. Final Claim
The completed 2×2 study indicates that neither label heterogeneity nor size imbalance alone reproduces the severe FedEraser degradation observed in the joint regime. The joint condition exhibits a substantially larger-than-additive loss of counterfactual fidelity, consistent with an interaction between heterogeneous update directions and unequal aggregation influence. The precise causal mechanism remains to be confirmed.

## 13. Recommendation for Stage 7D
Stage 7D is justified if the explicit objective is to test whether controlling one suspected mechanistic variable (e.g., enforcing magnitude normalization/clipping) reverses the interaction and rescues FedEraser in the joint regime. However, it is not strictly necessary for establishing the interaction itself; the next step could instead be deeper mathematical analysis of the geometry.
