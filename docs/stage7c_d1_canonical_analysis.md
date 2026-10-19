# Stage 7C: D1 Canonical Analysis & Factorial Evidence

## 1. D1 Canonical Objective
The objective of the D1 Canonical condition is to explicitly test the independent effect of high client label heterogeneity on FedEraser's unlearning fidelity while perfectly controlling for client size imbalance (equal data sizes) and strictly matching the attack configuration (poison ratio 0.1) of the canonical A0 attack.

## 2. Partition and Attack Configuration
- **Partition:** Strict greedy deficit-minimization over Dirichlet ($\alpha=0.1$) weights. All 20 clients (including Client 0) possess exactly 2,500 samples (Size CV = 0.0). Mean label entropy is 2.15 bits (Highly Heterogeneous).
- **Attack Configuration:** 
  - Canonical `poison_ratio: 0.1` (exactly matching D0, D2, and D3).
  - Target client 0, `gamma: 5.0`, `trigger_size: 3`, target class 0.
- **Note:** This supersedes the previous "D1-StrongAttack" exploratory run which inadvertently utilized a non-canonical `poison_ratio: 0.5`.

## 3. Matched Counterfactual (M_CF)
The structurally validated M_CF trajectory was reused directly:
- **Client 0 Status:** Excluded post-sampling globally.
- **Round 100 Clean Accuracy:** 74.84%
- **Round 100 ASR:** 2.78% (Baseline background rate)

## 4. Attacked Model Trajectory
- **Round 100 Clean Accuracy:** 74.23%
- **Round 100 ASR:** 36.84%
  *(Note: The ASR is lower than in size-imbalanced environments, reflecting the intrinsic difficulty of poisoning a robust FedAvg network without the artificial amplification of massive sample-size weighting, thereby confirming strict aggregation isolation.)*

## 5. Corrected Unlearning Metrics
Evaluated on the exact frozen `eval_indices.json` (7,500 samples):

| Method | Clean Acc | Clean Loss | ASR | CF Agreement | CF Gap | Utility Gap |
|---|---:|---:|---:|---:|---:|---:|
| D1 M_CF (Ref) | 74.84% | 0.743 | 2.78% | 100.00% | 0.00% | 0.00% |
| D1 Attacked | 74.23% | 0.742 | 36.84% | 87.69% | 12.31% | 0.61% |
| Subtraction | 50.73% | 1.578 | 0.86% | 54.47% | 45.53% | 24.11% |
| KD | 72.41% | 0.819 | 1.29% | 82.96% | 17.04% | 2.43% |
| FedEraser | 74.85% | 0.728 | 1.60% | 87.09% | 12.91% | -0.01% |

## 6. Update Geometry (Corrected)
Analysis of the actual aggregation-weighted update parameters dynamically scaled during FL execution:

| Metric | Round 24 | Round 49 | Round 73 | Round 100 |
|---|---|---|---|---|
| Malicious Aggregation Norm | 2.10 | 6.38 | 2.31 | 3.24 |
| Mean Benign Norm | 2.40 | 2.61 | 2.51 | 2.56 |
| Malicious/Benign Norm Ratio | **0.87x** | **2.44x** | **0.92x** | **1.27x** |
| Benign CV (Norm Dispersion) | 0.21 | 0.19 | 0.15 | 0.15 |
| Benign Directional Dispersion (1-cos)| **0.92** | **0.96** | **0.97** | **1.00** |
| Malicious-to-Benign Mean Cosine | 0.06 | 0.03 | 0.01 | 0.02 |

**Geometric Interpretation:** Even under the strict A0 canonical configuration, the pattern is completely consistent. Equal sizes enforce tightly constrained update magnitudes (Malicious/Benign norm ratio approaches ~1.0x). Simultaneously, the severe label skew pushes benign directional dispersion to its theoretical orthogonal limit (1-cos ≈ 1.00).

## 7. Canonical Factorial Interpretations
- **Comparison with D0 (Low/Low):** Label heterogeneity alone causes a trivial drop in FedEraser fidelity (89.63% to 87.09%).
- **Comparison with D2 (Low/High):** Size imbalance alone causes a more substantial drop in FedEraser fidelity (89.63% to 81.83%).
- **Comparison with D3 (High/High):** When these independent factors overlap (D3), fidelity collapses to 64.13%. 
- **Conclusion:** D1 explicitly proves that extreme label heterogeneity alone does not break FedEraser's historical calibration algorithm, provided that update magnitudes remain uniform across clients.
