# Canonical Unlearning Benchmark

## 1. Purpose
This document establishes the canonical, immutable benchmark specification for evaluating federated unlearning methods in the ISP project ("Removing Backdoors from Federated Models via Machine Unlearning"). All future unlearning methods (e.g. FedEraser, FedUP, etc.) must be evaluated strictly against this frozen framework to ensure fair, rigorous, and scientifically sound comparisons.

## 2. Benchmark Artifacts
The benchmark is anchored by the following frozen model states:

- **Starting Compromised Model (Teacher/Target):** 
  `results/attack_scaling_final_gamma5_20260909_015903`
  *(Contains the final Stage 3B global backdoored model and historical client updates).*
  
- **Counterfactual Reference (Gold Standard):** 
  `results/full_retraining_final_20260909_024627`
  *(Stage 4 full retraining from scratch excluding Client 0. To be used for evaluation comparison ONLY. Must never be used as an optimization target or teacher).*

- **Naive Historical Subtraction Baseline:** 
  `results/unlearning_5a_subtraction_20260909_131440`
  *(Stage 5A baseline).*

- **Historical Subtraction + KD Baseline:** 
  `results/unlearning_5b_kd_20260909_183734`
  *(Stage 5B baseline. Provides KD proxy/evaluation dataset splits).*

## 3. Threat Model and Forgotten Client
- **Target Client (Forgotten Client):** `Client 0`
- **Client Participation:** Client 0 participated in exactly 50 of the 100 federated rounds (14 clean, 36 attack).
- **Attack Configuration:** 3x3 white patch in the bottom-right corner. Target class = 0. Model-replacement scaling factor $\gamma = 5$.
- **Constraint:** Future methods must NOT alter the target client, attack configuration, or the historical participation trajectory.

## 4. Federated Data Partition
- **Total Clients:** 20
- **Partitioning:** IID
- **Samples per Client:** 2,500
- **Total Training Samples:** 50,000 (CIFAR-10)
- **Constraint:** Data partitions are strictly frozen.

## 5. Evaluation Protocol
To prevent evaluation leakage (since some unlearning methods like KD require proxy datasets), a strict dataset separation is enforced:

- **Proxy Split:** `proxy_indices.json` (2,500 images drawn from the CIFAR-10 test set with `seed=42`).
- **Common Evaluation Split:** `eval_indices.json` (The remaining 7,500 images).

**ALL future unlearning methods must use this exact 7,500-image Common Evaluation Split to report their final metrics.**

## 6. Metric Definitions
Every future method must report:
1. **Clean Accuracy:** Accuracy on the clean 7,500-image evaluation split.
2. **ASR (Attack Success Rate):** Accuracy on the triggered 7,500-image evaluation split (excluding naturally target-class images).
3. **Clean Loss:** Cross-entropy loss on the clean evaluation split.
4. **Prediction Agreement (Counterfactual Fidelity):** Percentage of identical predictions between the unlearned model and the Counterfactual Reference model on the evaluation split.

## 7. Resource Accounting Definitions
1. **Unlearning Time:** Total wall-clock time from the initiation of the unlearning method to the generation of the final unlearned model.
2. **Additional Communication:** Total client-server communication payload generated *exclusively* by the unlearning process. (Do not count the original FL training communication).
3. **Client Computation:** Any local training, distillation, or calibration performed by retained clients (Clients 1-19) during the unlearning process.
4. **Server Computation:** Any server-side optimization or distillation.
5. **Historical Storage:** Memory footprint of required existing artifacts (e.g. historical client updates).

## 8. Existing Baseline Results
Evaluated on the strict 7,500-image Common Evaluation Split:

| Method | Clean Accuracy | ASR | Unlearning Time | Additional Communication |
| :--- | :--- | :--- | :--- | :--- |
| **Backdoored (Stage 3B)** | 74.31% | 87.15% | — | — |
| **Historical Subtraction (Stage 5A)** | 10.53% | 0.00% | ~4.85 s | 0 Bytes |
| **Subtraction + KD (Stage 5B)** | 56.00% | 3.09% | ~15 s | 0 Bytes |
| **Full Retraining (Stage 4 - Reference)** | 74.48% | 3.02% | ~30 min | ~4.16 GB |

## 9. Rules for Future Methods
1. Do not use the Counterfactual Reference model (Stage 4) during optimization.
2. Do not use the Common Evaluation Split for any calibration or optimization.
3. Do not modify the existing historical client updates or checkpoints.
4. Report metrics consistently against the baselines defined above.
5. Record wall-clock time and communication explicitly.
