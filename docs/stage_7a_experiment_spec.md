# Stage 7A: Failure-Boundary Mapping for Federated Client Unlearning

## 1. Scientific Question
"Under what realistic federated conditions does the gap between efficient client-level unlearning and recovery of the client-free counterfactual become substantially larger?"

Specifically, we examine whether client heterogeneity (Non-IID data) and stronger adversarial influence create boundaries where current state-of-the-art methods (e.g., FedEraser) stop approximating the client-free counterfactual.

## 2. Hypothesis
**Working Hypothesis**: Greater client heterogeneity and stronger malicious influence make it harder for efficient unlearning methods to reconstruct the client-free counterfactual model. In Non-IID settings, a client's updates represent unique class features. Removing those updates via unlearning may cause catastrophic forgetting of benign features, and calibrating the remainder may fail if the remaining clients lack the heterogeneous data to correct the trajectory.

## 3. Experimental Design
A highly controlled, 4-condition screening matrix isolating data heterogeneity and attack strength, evaluating Historical Subtraction, Subtraction + KD, and FedEraser against an identical Matched Counterfactual.

### 3.1. Condition Definitions
1. **1_IID_Baseline**: IID data ($\alpha=\infty$), canonical attack ($\gamma=5$, start=30).
2. **2_Moderate_NonIID_Baseline**: Controlled benign-client heterogeneity ($\alpha=0.5$), canonical attack ($\gamma=5$, start=30).
3. **3_Strong_NonIID_Baseline**: Controlled benign-client heterogeneity ($\alpha=0.1$), canonical attack ($\gamma=5$, start=30).
4. **4_Strong_NonIID_StrongAttack**: Controlled benign-client heterogeneity ($\alpha=0.1$), stronger update-scaling stress ($\gamma=10$, start=30).

### 3.2. Partition Methodology (Controlled Benign-Client Heterogeneity)
- **Mechanism**: A custom `create_controlled_dirichlet_partitions` mechanism that explicitly pins the malicious client (Client 0) to exactly its baseline IID allocation (2,500 evenly distributed samples). It then partitions the remaining 47,500 samples among benign clients 1-19 using standard unconstrained Dirichlet partitioning (`np.random.dirichlet`).
- **H2 Terminology**: H2 represents a strong controlled benign-client non-IID regime produced by unconstrained Dirichlet allocation, resulting in substantial label skew and client-size imbalance, while Client 0's local clean data is held fixed.
- **Resulting Entropy**:
  - H0: IID (Avg Benign Class Entropy: 2.301)
  - H1: $\alpha=0.5$ (Avg Benign Class Entropy: 1.714)
  - H2: $\alpha=0.1$ (Avg Benign Class Entropy: 0.909)

### 3.3. Attack Methodology
- **A0_Baseline**: The established Stage 3B canonical attack. Scaling factor $\gamma=5$, Trigger size 3x3, `attack_start=30`, `attack_end=100`.
- **A1_Stronger**: The stronger stress condition. $\gamma=10$, Trigger size 3x3, `attack_start=30`, `attack_end=100`.
- **Rationale**: Increasing the scale factor explicitly amplifies the malicious geometric projection during aggregation without altering the trigger semantic complexity, attack duration, or poisoning ratio, providing a clean test of attack strength robustness.

### 3.4. Controls and Counterfactuals
For every data condition, three models are trained:
- **Attacked Model**: Standard federated training with Client 0 injecting backdoors starting at round 30.
- **No-Attack Control**: Clean federated training (verifies that the underlying FL task can still converge under the specific Non-IID regime).
- **Matched Counterfactual ($M_{CF}$)**: Clean training with Client 0 explicitly dropped post-sampling. Preserves the exact stochastic trajectory of the 19 honest clients, forming the absolute gold standard for unlearning fidelity.

## 4. Unlearning Methods
- Historical Subtraction (Linear target update removal).
- Historical Subtraction + Knowledge Distillation.
- FedEraser (Trajectory replay and calibration).

## 5. Storage Projection and Required Code Modifications
- **Free Space Available**: ~11.0 GB
- **Standard Attacked Run**: ~2.3 GB (100 global checkpoints + 1000 client updates).
- **Execution Projection**: 4 Attacked Runs + 4 No-Attack Runs + 4 $M_{CF}$ Runs = 12 total runs. (27.6 GB required using standard saving).
- **Mitigation Requirement**: Before execution in Phase B, `run_federated.py` MUST be modified to support `--skip_client_updates` (for No-Attack and $M_{CF}$ runs) and `--save_final_only` (to prevent storing 100 global checkpoints). With these optimizations, total required storage drops to ~9.6 GB, fitting safely within disk constraints.

## 6. Execution Order
- **Phase A**: Partition Generation and Validation (Completed).
- **Phase B**: Training Screen (Execute the 12 models with optimized storage).
- **Phase C**: Unlearning Execution and Consolidated Metric Evaluation.
