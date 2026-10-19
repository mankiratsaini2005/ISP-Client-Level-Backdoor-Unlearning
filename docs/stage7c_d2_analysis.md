# Stage 7C: D2 Downstream Evaluation

## 1. D2 Objective
The primary objective of the D2 condition is to isolate the effect of client-size imbalance on federated unlearning efficacy, particularly for FedEraser. D2 is characterized by "low label heterogeneity" (IID distributions) but "high size imbalance" (matching H2 sizes).

## 2. D2 Partition
The exact D2 partition was verified. Client 0 retains exactly 2,500 samples (its canonical fixed size), while the benign clients exhibit sizes ranging from ~400 to ~900 samples, precisely matching the size heterogeneity of the highly-skewed H2 partition.

## 3. D2 Attack
The D2 attacked global model (canonical A0 parameters: gamma=5.0, Client 0, Trigger 3, Class 0) achieved:
- **Clean Accuracy**: 77.61%
- **ASR**: 72.48%
These values strictly match the validated D2 recovery run.

## 4. True M_CF
The true Matched Counterfactual (M_CF_D2) was generated successfully using the exact D2 partition and sampling history, with Client 0 explicitly dropped post-sampling. M_CF_D2 structural checks passed (Client 0 did not influence aggregation). 
- **Clean Accuracy**: 77.40%
- **ASR**: 2.66%

## 5-7. Unlearning Methods (Historical Subtraction, KD, FedEraser)
Unlearning was applied against the D2 Attacked model targeting Client 0. M_CF_D2 was reserved strictly for evaluation.

| Method | Clean Acc | ASR | CF Agreement | CF Gap | KL |
|---|---:|---:|---:|---:|---:|
| D2 M_CF (reference) | 77.40% | 2.66% | 100.00% | 0.00% | 0.000 |
| Subtraction | 10.00% | 0.00% | 10.49% | 67.40% | 55.433 |
| KD | 58.24% | 4.06% | 64.63% | 19.16% | 1.830 |
| FedEraser | 72.95% | 1.89% | 81.83% | 4.45% | - |

*(Note: KL was not recorded for FedEraser in the final audit, relying on Agreement instead)*

## 8. Counterfactual Fidelity
FedEraser achieved a moderately high CF fidelity (81.83% agreement) with the true M_CF. This is substantially better than the catastrophic failures often observed in extreme label skew, yet noticeably imperfect. KD and Subtraction performed poorly, as expected in deep unlearning without iterative reconstruction.

## 9. Clean Utility
FedEraser restored clean utility to 72.95%, reflecting a relatively small CF Gap (4.45%) from the canonical clean trajectory (77.40%). 

## 10. Attack Suppression
Both FedEraser (ASR 1.89%) and KD (ASR 4.06%) successfully suppressed the BadNets backdoor down to baseline misclassification levels, eliminating the attack.

## 11. Update Geometry
Analysis of the malicious vs. benign updates at Round 24, 49, 73, and 100 reveals:
- The scaled malicious update norm remains highly dominant over the benign average (ratio ~7.6x early, rising to ~45x midway, and ~24.6x late).
- The benign directional dispersion (1 - mean pairwise cosine) remains extremely high (~0.95 - 0.97).
- The coefficient of variation (CV) for benign norms is around ~0.45 - 0.58. 
This geometry confirms that the malicious attack update dominates the aggregation space, but the benign clients are highly dispersed in direction.

## 12. Calibration Error
Native un-modified FedEraser code does not support intermediate calibration error dumping. Therefore, while final fidelity is available, the exact per-client, per-round calibration residual could not be extracted without modifying the unlearning algorithm (which is prohibited under the strict rules).

## 13. D2 vs D0
D0 (low label heterogeneity + low size imbalance) typically yields near-perfect FedEraser reconstruction. D2 (low label heterogeneity + *high* size imbalance) yields an 81.83% CF Agreement. The drop from near-perfect to ~82% highlights that size heterogeneity alone actively degrades FedEraser's historical calibration efficacy.

## 14. D2 vs D3/H2
D3/H2 (high label heterogeneity + high size imbalance) exhibits massive calibration degradation and often poor utility. Because D2 recovers 72.95% clean accuracy and 81.83% fidelity, it strongly outperforms D3. This suggests that while size imbalance (D2) degrades unlearning, label heterogeneity (D3) is the primary driver of catastrophic unlearning failure.

## 15. Interpretation
The results suggest that client size imbalance creates unequal weighting in historical calibration, leading to moderate divergence from the true counterfactual. However, because the underlying data is IID (low label heterogeneity), the clients still share roughly similar global optimization directions, preventing complete divergence.

## 16. Threats to Validity
- Calibration error could not be directly quantified due to code immutability.
- Causal attribution between size imbalance and FedEraser degradation is supported but not definitively proven without completing the D1 (high label + equal size) quadrant.

## 17. Evidence for/against size-imbalance hypothesis
**Classification: B. Moderate evidence.**
The D2 condition demonstrates that introducing size imbalance into an otherwise IID setting introduces measurable degradation in FedEraser fidelity (falling to ~82%). Size imbalance does negatively impact historical reconstruction.

## 18. Recommendation for D1
Proceed with Stage 7C D1 (high label heterogeneity + equal size). Evaluating D1 is now the critical missing link to definitively apportion the relative blame between size imbalance and label skew for FedEraser's degradation.
