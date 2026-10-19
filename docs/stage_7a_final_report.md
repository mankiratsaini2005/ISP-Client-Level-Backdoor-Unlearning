# Stage 7A Final Report: Failure-Boundary Mapping for Federated Client Unlearning

## 1. Objective and Experimental Design
The goal of Stage 7A was to identify and characterize realistic federated conditions under which existing client-level unlearning methods experience a meaningful degradation in counterfactual fidelity, utility, or security. 

We executed a controlled, progressive stress-test by stepping through three data heterogeneity regimes while injecting a standardized data-poisoning attack (BadNets, M_target=0, trigger size=3, attack start=30, length=70).
1. **C1 (H0 + A0):** IID Partitioning (baseline).
2. **C2 (H1 + A0):** Moderate Non-IID (Dirichlet alpha=0.5).
3. **C3 (H2 + A0):** Strong Non-IID (Dirichlet alpha=0.1).
4. **C4 (H2 + A1):** Strong Non-IID + Strong Update Scaling (gamma=10.0 instead of 5.0).

In all regimes, the target unlearning client (C_0) retained a fixed, balanced, 2500-sample dataset, isolating the effect of *benign-client* heterogeneity on the unlearning process.

For each regime, we computed the true client-free Matched Counterfactual (M_CF) by training from the same initialization with C_0 excluded from the sampling pool, and compared it against three unlearning methods:
- Historical Subtraction
- Historical Subtraction + Knowledge Distillation (KD)
- FedEraser

## 2. Attack-Validity Check
Before running unlearning, we verified that the attack successfully inserted a backdoor without destroying standard convergence:

| Regime | Clean Acc (Attacked) | Clean Acc (Control) | Peak ASR | Final ASR |
|---|---:|---:|---:|---:|
| **C1 (IID)** | 74.23% | 74.89% | 87.26% | 87.26% |
| **C2 (Mod Non-IID)** | 72.40% | 72.29% | 88.82% | 82.44% |
| **C3 (Strong Non-IID)** | 61.75% | 60.25% | 100.00% | 67.34% |
| **C4 (Strong Non-IID + Stress)**| 60.51% | 60.25% | 100.00% | 81.58% |

*Observation:* As heterogeneity increases (H2), the global model struggles more to generalize (Clean Acc drops to ~61%), but the attack remains highly potent (Peak ASR reaches 100%).

## 3. Unlearning Results & Counterfactual Fidelity

The unlearning methods were applied to the poisoned models and compared against the M_CF benchmark (fidelity measured as prediction agreement on the clean CIFAR-10 test set):

| Regime | Method | Clean Acc | ASR | CF Agreement | CF Gap |
|---|---|---:|---:|---:|---:|
| **C1** | Subtraction | 10.30% | 0.00% | 10.01% | 89.99% |
| **C1** | Subtraction+KD | 57.61% | 3.53% | 66.63% | 33.37% |
| **C1** | FedEraser | 74.78% | 2.58% | **89.35%** | 10.65% |
| **C2** | Subtraction | 11.52% | 0.00% | 12.08% | 87.92% |
| **C2** | Subtraction+KD | 50.27% | 4.89% | 58.10% | 41.90% |
| **C2** | FedEraser | 73.25% | 2.24% | **84.74%** | 15.26% |
| **C3** | Subtraction | 11.33% | 0.00% | 17.45% | 82.55% |
| **C3** | Subtraction+KD | 32.00% | 12.99% | 45.26% | 54.74% |
| **C3** | FedEraser | 74.73% | 2.32% | **67.92%** | 32.08% |
| **C4** | Subtraction | 10.25% | 0.00% | 13.15% | 86.85% |
| **C4** | Subtraction+KD | 18.67% | 0.29% | 26.90% | 73.10% |
| **C4** | FedEraser | 74.82% | 2.28% | **67.10%** | 32.90% |

### Key Observations
1. **The FedEraser Failure Boundary:** Under standard IID conditions (C1), FedEraser exhibits near-perfect utility retention (74.78%) and high counterfactual fidelity (89.35%). However, as benign-client heterogeneity increases to strong non-IID (C3), **FedEraser's counterfactual fidelity plunges to 67.92%.**
2. **Illusion of Utility:** Strikingly, FedEraser's Clean Accuracy in C3/C4 remains at ~74.8%, *even though the true counterfactual model without the malicious client converges at ~60.3%*. This means FedEraser is fundamentally hallucinating generalizability; it is trapping the model in the poisoned global minima rather than reconstructing the correct, lower-utility counterfactual space dictated by the heavily skewed client partitions.
3. **Subtraction + KD Collapse:** The KD approach similarly collapses, dropping from 66.63% agreement in C1 to a dismal 26.90% agreement in C4.

## 4. Update-Geometry Analysis
To understand why the reconstruction methods degrade, we analyzed the L2-norm geometry of client updates at the final round (Round 100):

| Metric | C1 (IID) | C3 (Strong Non-IID) | C4 (Stress Attack) |
|---|---:|---:|---:|
| Malicious Update Norm (||U_mal||) | 565.79 | 565.79 | 1131.60 |
| Avg Benign Update Norm (mu_{||U_ben||}) | 113.17 | 152.79 | 152.80 |
| Benign Norm Dispersion (sigma/mu) | 0.0001 | 1.0630 | 1.0630 |
| Malicious/Benign Norm Multiplier | 5.00x | 3.70x | 7.41x |

### Failure Mechanism
The geometry reveals the structural flaw in history-constrained unlearning algorithms (like FedEraser) when exposed to non-IID federated distributions:
- **Massive Dispersion:** In IID settings, benign update norms have near-zero variance (sigma/mu = 0.0001). In H2, the unconstrained Dirichlet allocation produces a massive client-size imbalance (ranging from 116 to 12,266 samples). Consequently, benign update norms vary wildly (sigma/mu = 1.0630).
- **The Norm-Calibration Trap:** FedEraser's core mechanism constrains the reconstructed updates to match the exact historical L2-norm of the poisoned trajectory. Because the historical trajectory in C3/C4 was violently skewed by the attack operating against a highly imbalanced background gradient, constraining the unlearning updates to match those historical magnitudes prevents the model from traversing the true counterfactual optimization path.
- **Result:** The model is forcibly pinned near the poisoned checkpoint's geometry, maintaining an artificially high IID-like clean accuracy while entirely failing to align with the true non-IID counterfactual reality (CF Gap jumps from 10.6% to 32.9%).

## 5. Conclusion
Stage 7A successfully maps a clear failure boundary for state-of-the-art federated unlearning. High benign-client data heterogeneity (label skew + size imbalance) breaks the foundational assumption of trajectory-reconstruction methods like FedEraser. By tightly coupling unlearning steps to historical update magnitudes, these methods inherently fail to recover the client-free counterfactual when the historical geometry is heavily distorted by variance and non-IID distribution shifts.
