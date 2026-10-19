# Post-B1 Research Review & Strategic Roadmap

## A. Phase B-1 Results Analysis
Phase B-1 tested three unlearning methods (Subtraction, KD, FedEraser) against three identical-magnitude, temporally shifted malicious influence patterns (Early, Middle, Late) originating from Client 0. 

**Robust Observations:**
- **Linear Subtraction** universally destroyed model utility (dropping accuracy by ~40%) and failed to recover counterfactual fidelity (only ~35-41% agreement with $M_{CF}$). It exhibited measurable timing sensitivity, performing slightly worse in fidelity when attempting to unlearn early updates.
- **Knowledge Distillation** improved utility and fidelity over Subtraction but hit a ceiling (fidelity ~74-78%). Since KD distills from the compromised model, it inherently anchors the student to the backdoored optimization path.
- **FedEraser** was highly effective at recovering clean utility, suppressing ASR to baseline false-positive levels, and achieving the highest fidelity (~89%). Under our current experimental parameters, FedEraser exhibited highly stable fidelity across all three temporal windows.

## B. Deltas and Gap Analysis

*(Reference $M_{CF}$: Clean Acc 74.76%, ASR 2.32%)*

| Metric | Subtraction (E/M/L) | KD (E/M/L) | FedEraser (E/M/L) |
| :--- | :--- | :--- | :--- |
| **Utility Gap to $M_{CF}$** | -41.4% / -36.0% / -37.0% | -7.0% / -7.6% / -9.7% | -0.7% / -0.1% / +0.3% |
| **Fidelity Gap to $M_{CF}$** | 64.7% / 59.2% / 58.7% | 21.7% / 22.2% / 26.0% | 11.0% / 11.5% / 11.0% |
| **ASR Suppression** | To ~0.1% | To ~2.5% | To ~2.6% |
| **Runtime Efficiency** | ~7s | ~6s | ~1630s (27 min) |
| **Timing Sensitivity (Max Delta)**| 6.04% fidelity variation | 4.30% fidelity variation | 0.56% fidelity variation |

## C. Robust Observations vs. Hypotheses
*   **Robust Observation**: Under the single-seed configuration tested, FedEraser successfully calibrates out targeted client influence while maintaining trajectory stability regardless of temporal placement. Subtraction does not.
*   **Hypothesis**: FedEraser is universally invariant to attack timing. *(Phase B-1 does not prove this; it only proves stability under a specific scaling/data regime).*
*   **Hypothesis**: Early attacks induce greater path-dependence. *(Supported by Subtraction data, but fully neutralized by FedEraser).*

## D. What Phase B-1 Ruled Out
Phase B-1 ruled out the hypothesis that **temporal placement alone** represents a fundamental, catastrophic failure mode for existing state-of-the-art trajectory-aware unlearning (FedEraser) in an IID, 20-client, 100-round setting.

## E. What Phase B-1 Did NOT Rule Out
It did not rule out failure modes arising from:
- Multi-seed statistical variance.
- High-intensity backdoor scaling ($\gamma > 5$).
- Non-IID data distributions (where client exclusion creates severe feature gaps).
- Maliciously adaptive attackers.

## F. Strongest Remaining Scientific Failure Modes
1.  **The Non-IID Feature Collapse**: If client data is highly heterogeneous, calibrating their absence may cause catastrophic forgetting of genuine features, destroying utility.
2.  **Adaptive Security / Unlearning Sabotage**: An attacker aware of the unlearning mechanism might inject updates specifically designed to evade calibration or weaponize the unlearning step itself to induce a secondary attack (e.g., targeted forgetting of a specific class).
3.  **Scale and Capacity**: The gap (~11%) between FedEraser and $M_{CF}$ might widen uncontrollably if the model capacity is smaller or the task is harder.

## G. Necessity of Multi-Seed Runs
To make strong claims about *timing sensitivity*, repeating experiments across multiple random seeds is strictly necessary. Given the ~0.56% variance in FedEraser fidelity across Early/Middle/Late, we cannot statistically claim whether the difference is zero or non-zero without computing standard deviations across seeds. However, since the effect size is clearly negligible compared to Subtraction's 60% gap, investing compute into multi-seed temporal sweeps offers a poor scientific ROI compared to pivoting to a starker failure mode.

## H. Evaluation of Standard Sweeps
*   **Attack Strength ($\gamma$) Sweep**: Standard due diligence. High $\gamma$ might break FedEraser's first-order Taylor approximation used during calibration.
*   **Participation Sweep**: Investigates how heavily ingrained the client is.
*   **IID $\rightarrow$ Non-IID**: Highly valuable. Non-IID breaks FedEraser's assumption that retained clients can effectively estimate the global direction.

## I. The Emerging Security Question: Adaptive Unlearning Attacks
Can an attacker adapt to the unlearning process?
Current unlearning literature largely treats unlearning as a post-hoc "clean up" operation against *static* backdoor attacks designed for standard FL. An adaptive attacker could:
1.  **Entanglement Attacks**: Design an update that maximally entangles the backdoor trigger with benign features belonging to other clients, forcing the unlearning calibration step to destroy global utility.
2.  **Calibration Evasion**: Design an update that looks benign to the unlearning calibration's first-order approximation but diverges sharply at higher orders, ensuring the backdoor survives FedEraser.
3.  **Sabotage / Secondary Poisoning**: Inject a backdoor that remains dormant *until* the unlearning step is executed, using the calibration gradient inversion to activate the payload.

## J. Research Value Comparison

| Direction | Focus | Scientific Value | Risk of Duplication |
| :--- | :--- | :--- | :--- |
| **1. Path-Dependence/Timing** | Extending B-1 with seeds/scales. | Low-Medium (Incremental) | High (Well-studied phenomenon) |
| **2. Failure Boundaries** | Non-IID, extreme $\gamma$, scale. | Medium (Good benchmarking) | High (Standard evaluation procedure) |
| **3. Adaptive Attacks** | Sabotaging the unlearning process. | **Very High** (Novel threat model) | Low (Under-explored in Federated contexts) |
| **4. Novel Unlearning Method** | Building a new defense. | Medium | High (Crowded space, hard to justify without a stark baseline failure) |

## K. Ranking the Directions

1.  **Study Adaptive Attacks Against Unlearning**: (Novelty: High, Importance: High, Feasibility: High, Compute: Medium, Pub. Potential: High, Duplication Risk: Low).
2.  **Study Failure Boundaries (Non-IID)**: (Novelty: Medium, Importance: Medium, Feasibility: High, Compute: High, Pub. Potential: Medium, Duplication Risk: High).
3.  **Study Path-Dependence (Multi-seed)**: (Novelty: Low, Importance: Low, Feasibility: High, Compute: Very High, Pub. Potential: Low, Duplication Risk: High).
4.  **Develop Novel Unlearning Method**: (Novelty: Medium, Importance: Medium, Feasibility: Low-Medium, Compute: High, Pub. Potential: Medium, Duplication Risk: Very High).

## L. Recommended Next Research Direction
**Pivot to Direction 3: Adaptive Attacks Against the Unlearning Process.**
Instead of searching for marginal failures in standard settings, we should shift the threat model. FedEraser assumes an attacker who optimizes for ASR during normal training. If the attacker optimizes to survive or weaponize the FedEraser unlearning process itself, the current "state-of-the-art" may fail catastrophically. This transitions the project from a defensive benchmark into an offensive security discovery, offering significantly higher scientific novelty.

## M. Exact Experiment Sequence Recommendation
If this direction is approved, the exact next sequence should be:

**Stage 7A: Entanglement Attack Formulation**
- Design an adversarial objective that maximizes the cosine similarity between the backdoor gradient and the average benign gradient of the dataset (e.g., using a proxy set).

**Stage 7B: Adaptive Training**
- Implement this objective as a custom local training loop for Client 0.
- Execute a single federated training run using this new adaptive attacker.

**Stage 7C: Unlearning Sabotage Test**
- Run FedEraser on the adaptively poisoned model.
- Evaluate if ASR survives, or if Clean Utility is catastrophically destroyed during calibration.
- Compare against the standard Stage 3B attack results.
