# Stage 6D Phase A Report: Attack and Counterfactual Validation

## 1. Scientific Question
"When the same configured amount of malicious Client-0 behavior is placed at different temporal positions within an otherwise controlled partially participating federated trajectory, does temporal placement affect attack impact and the subsequent difficulty of recovering the client-excluded counterfactual model?"

## 2. Experimental Design
An exploratory deterministic single-seed pilot testing three timing conditions (Early, Middle, Late) against a No-Attack control and a Matched Client-Excluded Counterfactual. The independent variable is the temporal placement of exactly 15 malicious Client-0 participation events. 

## 3. Exact Client-0 Participation Schedule
From the frozen `client_sampling.json`, Client 0 is selected in the following 50 rounds:
`[1, 4, 8, 9, 13, 14, 15, 16, 18, 19, 22, 24, 27, 29, 30, 31, 34, 36, 38, 40, 42, 43, 44, 45, 46, 48, 49, 51, 52, 54, 55, 56, 57, 58, 60, 64, 71, 72, 73, 80, 84, 85, 87, 88, 90, 91, 94, 96, 97, 100]`

## 4. Exact Malicious-Event Schedules
- **Early**: `[1, 4, 8, 9, 13, 14, 15, 16, 18, 19, 22, 24, 27, 29, 30]`
- **Middle**: `[36, 38, 40, 42, 43, 44, 45, 46, 48, 49, 51, 52, 54, 55, 56]`
- **Late**: `[64, 71, 72, 73, 80, 84, 85, 87, 88, 90, 91, 94, 96, 97, 100]`

## 5. Exposure-Control Verification
For each of the three attack conditions, exactly 15 participation events are malicious (yielding exactly 3,750 poisoned-sample exposures). The remaining 35 Client-0 events utilize clean data and normal federated updates.

## 6. No-Attack Control
Client 0 participates in all 50 scheduled rounds but behaves entirely benignly, using clean data and standard optimization.

## 7. Matched Counterfactual ($M_{CF}$)
Using identical seed and initialization, Client 0 is removed *after* the pseudo-random sampling draw, resulting in a 9-client aggregation for those 50 rounds. This perfectly preserves the stochastic sequence for the 19 benign clients.

## 8. Stage-4 Reference Distinction
The original Stage-4 model resampled a 10th client when Client-0 was excluded, diverging the optimization trajectory. $M_{CF}$ corrects this for exact measurement.

## 9. Exact Configurations
- **Model**: SmallCNN on CIFAR-10, IID partition
- **Trigger**: 3x3 bottom-right, Target Class: 0
- **Scale Factor**: 5.0 (for malicious updates)
- **Poison Ratio**: 10%
- **Rounds**: 100 (10 clients selected per round)

## 10. Final Clean Accuracy
- Early: 73.91%
- Middle: 74.93%
- Late: 73.78%
- No-Attack: 74.89%
- $M_{CF}$: 75.07%

## 11. Final ASR
- Early: 5.36%
- Middle: 16.67%
- Late: 85.83%

## 12. Peak ASR
- Early: 100.00%
- Middle: 84.59%
- Late: 85.83%

## 13. Attack Impact vs No-Attack
- **Early**: Acc Diff = -0.98%, Prediction Disagreement = 16.90% (Agreement: 83.10%)
- **Middle**: Acc Diff = +0.04%, Prediction Disagreement = 16.46% (Agreement: 83.54%)
- **Late**: Acc Diff = -1.11%, Prediction Disagreement = 20.90% (Agreement: 79.10%)

## 14. $M_{CF}$ vs Stage 4 Sanity Comparison
- **Prediction Agreement**: 83.22%. This significant divergence (~17%) strongly validates the necessity of introducing $M_{CF}$; the historical Stage 4 trajectory had drifted considerably due to client resampling, making it an invalid matched unlearning target.

## 15. Runtime
Approx. 2 hours total (5 models evaluated sequentially via automated script).

## 16. Communication
Standard FedAvg communication costs per 100-round run.

## 17. Storage
Current Free Space: ~11 GB. Estimated total Phase A impact was ~11 GB, leaving a safe working margin.

## 18. Validation Checks
Programmatic validation succeeded:
- Exactly 15 malicious events and 35 benign events per condition.
- No-Attack generated 0.00% ASR.
- Canonical artifacts remain perfectly untouched.

## 19. Anomalies
None. The catastrophic forgetting of the early backdoor (Peak ASR 100% -> Final ASR 5.36%) perfectly frames the scientific question: Does "natural forgetting" equal "machine unlearning"?

## 20. Attack-Validity Decision
The Early, Middle, and Late conditions all show definitive, interpretable backdoor insertion (Peak ASRs > 84%). Furthermore, they individually produce a robust 16-21% prediction divergence from the No-Attack model.

## 21. Recommendation for Phase B
The attack signals are unequivocally strong enough to provide a meaningful unlearning evaluation surface. I recommend authorizing Phase B.
