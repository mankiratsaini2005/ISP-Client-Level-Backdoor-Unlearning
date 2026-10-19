# FedEraser Adaptation Specification

## 1. Purpose
This document provides a detailed specification for implementing a faithful adaptation of the "FedEraser" federated unlearning algorithm (Liu et al., IWQoS 2021) into the ISP project repository. It resolves critical methodological mappings between the paper's theoretical framework and our partial-participation, PyTorch-based Federated Learning setup. 

## 2. Original FedEraser Method
FedEraser is an iterative unlearning algorithm that reconstructs a global model by substituting the unlearned client's influence with calibrated updates from retained clients. It traces the dependency of the target client's data across historical FL rounds. Rather than retraining from scratch, the server broadcasts intermediate reconstructed models to historically participating retained clients. These clients perform "calibration training" to produce new update *directions*. The server then scales these new directions using the *magnitudes* of the historically stored updates to form calibrated updates, which are aggregated to advance the reconstructed global model.

## 3. Exact Mathematical Formulation
Let $U_k^{(t)}$ be the original stored historical update from client $k$ at round $t$.
Let $\hat{U}_k^{(t)}$ be the new calibration update obtained by training on the intermediate reconstructed model $M'_{t-1}$.
The calibrated update $\tilde{U}_k^{(t)}$ is defined by scaling the normalized direction of the new calibration update by the L2 norm (magnitude) of the historical update:
$$\tilde{U}_k^{(t)} = ||U_k^{(t)}||_2 \times \frac{\hat{U}_k^{(t)}}{||\hat{U}_k^{(t)}||_2}$$
where $||\cdot||_2$ denotes the L2 norm of the flattened parameter vector.

## 4. Our Canonical FL Setting
- **Total Clients:** 20
- **Data per Client:** $N_k = 2500$ (IID)
- **Selection per Round:** $|S_t| = 10$ (Partial participation)
- **Local Epochs:** $E_{local} = 2$
- **Target Client:** Client 0 (participated in 50 of 100 rounds)

## 5. Paper-to-Repository Mapping
- **Initial/Global Model:** `results/attack_scaling_final_gamma5_20260909_015903/checkpoints/global_round_000.pt` to `100.pt`
- **Historical Client Updates:** `results/attack_scaling_final_gamma5_.../client_updates/round_{t}/client_{k}.pt`
- **Client Partitions:** `results/attack_scaling_final_gamma5_.../client_partitions.json`
- **Participation History:** `results/attack_scaling_final_gamma5_.../client_sampling.json`
- **Target Client:** Client 0
- **Local Training Routine:** `src.training.local.LocalTrainer`
- **Aggregation Routine:** Standard FedAvg (to be adapted for calibration)
- **Evaluation Routine:** `src.data.backdoor.create_asr_dataloader` and evaluation functions.

## 6. Partial-Participation Analysis
FedEraser aggregates updates based on the ratio of data held by the participating clients. In our partial-participation setting:
**Distinction between Client Weights and Participation Normalization:**
Each client $k$ has $N_k = 2500$ samples. Standard FedAvg normalizes these weights over the participating set $S_t$: $p_k = \frac{N_k}{\sum_{j \in S_t} N_j}$. Originally, with $|S_t| = 10$, $p_k = 0.1$.
**Adaptation Analysis:**
- **Option A (Retain 0.1):** Causes the total aggregation weight to sum to 0.9 in rounds where Client 0 is removed, artificially decaying the global learning rate.
- **Option B (Renormalize):** FedEraser explicitly defines the aggregation weight as $p_k = \frac{N_k}{\sum_{j \in S_t \setminus \{0\}} N_j}$. Since $N_k$ is constant, this simplifies to $p_k = \frac{1}{|S_t \setminus \{0\}|}$.
**Decision:** We will use **Option B**. If Client 0 participated, $|S_t \setminus \{0\}| = 9$, and $p_k = 1/9$. If Client 0 did not participate, $|S_t \setminus \{0\}| = 10$, and $p_k = 1/10 = 0.1$. This rigorously follows FedEraser's aggregation logic while adapting to partial participation.

## 7. Reconstruction Schedule & Calibration Set
The paper dictates that reconstruction must begin at the *first* round where the target client's data influenced the global model. 
Since our participation history (`client_sampling.json`) shows Client 0 participated in Round 1, the dependency trace requires reconstructing from $t=1$ to $t=100$. Given our full historical artifact availability, we propose instantiating this complete 100-round reconstruction.

**Calibration Set per Round:**
The paper states that the retained clients from the historical round perform the calibration.
- **Case A (Client 0 did NOT participate):** The historical set $S_t$ calibrates. (10 clients).
- **Case B (Client 0 DID participate):** The retained set $S_t \setminus \{0\}$ calibrates. (9 clients).

## 8. Calibration Training
To generate the new direction $\hat{U}_k^{(t)}$ that is directly comparable to the historical update $U_k^{(t)}$, the local calibration training must mirror the original local training exactly.
- **Calibration Epochs:** $E_{cali} = E_{local} = 2$.
- **Optimizer:** SGD (LR=0.01, Momentum=0.9, WD=5e-4).

## 9. Update Calibration Mathematics
- **L2 Norm Computation:** The norm $||U_k^{(t)}||_2$ is computed globally by flattening all floating-point trainable tensors into a single 1D vector. This scalar magnitude is applied across the identically flattened calibration update $\hat{U}_k^{(t)}$.
- **Zero-Norm Handling:** If $||\hat{U}_k^{(t)}||_2 < 10^{-8}$, the scaled update defaults to $\hat{U}_k^{(t)}$ to prevent division-by-zero NaNs.

## 10. BatchNorm / state_dict Adaptation
**PyTorch/BatchNorm Implementation Adaptation:**
The FedEraser magnitude/direction scaling equation is designed for gradient updates on trainable parameters. Applying L2 norm scaling to statistical buffers like `running_var` mathematically corrupts the variance. 
**Rule:** Only tensors of type `float` that have `requires_grad=True` in the original model definition (weights and biases) will undergo norm/direction calibration. Statistical buffers (`running_mean`, `running_var`, `num_batches_tracked`) generated by the calibration training will be aggregated using standard FedAvg without magnitude scaling.

## 11. Evaluation & Fair-Comparison Rules
- **Stage 4 Isolation:** The Stage 4 counterfactual model is used ONLY for final evaluation comparison. It must not provide parameters, logits, or training targets to FedEraser.
- **Target-Client Information:** FedEraser receives ONLY the instruction to "forget Client 0". It does not receive attack labels, trigger locations, or poisoned sample indices.
- **Common Evaluation:** The final FedEraser model will be evaluated strictly on the frozen `eval_indices.json` (7,500 common images) to match the Stage 5B benchmark.

## 12. Resource Accounting
We will explicitly measure and distinguish original FL cost from unlearning cost:
- **Unlearning Wall-Clock Time:** Total time from $t=1$ to $t=100$ reconstruction.
- **Additional Communication:** Number of bytes transmitted between server and calibrating clients *during* unlearning. (Original FL communication is not counted).
- **Client Computation:** Total number of local calibration epochs executed across all retained clients.
- **Historical Storage:** Disk space consumed by the Stage 3B historical updates.

*Note: FedEraser is expected to incur substantial additional client/server computation because its published procedure includes retained-client calibration for every affected round. Exact runtime and communication will be measured experimentally.*

## 13. Metrics
1. **Clean Accuracy** (7,500-image evaluation set)
2. **Clean Loss** 
3. **ASR** (7,500-image triggered evaluation set)
4. **Counterfactual Fidelity:** Prediction agreement with the Stage 4 Counterfactual Reference.
5. **Efficiency:** Unlearning runtime, additional communication, client computation.

## 14. Tests
1. **Target client exclusion:** Assert Client 0 is never in $S_t \setminus \{0\}$.
2. **Retained client correctness:** Assert only clients from the original $S_t$ calibrate.
3. **Weight normalization:** Assert aggregation weights sum to 1.0.
4. **Zero-norm handling:** Assert no NaNs when $||\hat{U}||_2 = 0$.
5. **BatchNorm preservation:** Assert buffers do not undergo magnitude scaling.

## 15. Known Deviations from Published Method
1. **BatchNorm Bypassing (Implementation Adaptation):** FedEraser equations assume pure gradient updates. We adapt this for PyTorch by exempting statistical buffers from magnitude scaling to prevent mathematical corruption.

## 16. Risks
- **Calibration Accumulation Drift:** With 100 rounds of reconstruction, small errors in the calibration direction $\hat{U}$ may compound, potentially causing the reconstructed model to diverge significantly from the true counterfactual Stage 4 reference.

## 17. Final Implementation Decision
**OPTION B: FedEraser is implementable with clearly defined methodological adaptations.** (Specifically, the partial-participation aggregation renormalization and the PyTorch BatchNorm buffer exemption).
