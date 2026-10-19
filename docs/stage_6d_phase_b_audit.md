# Stage 6D Phase B: Design Audit and Execution Plan

## A. CURRENT IMPLEMENTATION AUDIT

1. **Historical Subtraction (`unlearn_stage5a.py`)**: 
   - **Mechanism**: Reads `client_sampling.json`, isolates the target client's participations, loads their scaled historical updates from `client_updates/`, accumulates them via summation, and subtracts them from the final global model.
   - **Status**: Completely parameterized and path-agnostic. It can be reused for Early, Middle, and Late without any code changes simply by passing the respective `--source_dir`.

2. **Historical Subtraction + KD (`unlearn_stage5b.py`)**:
   - **Mechanism**: Initializes a student with the Subtraction model, a teacher with the Backdoored model, and distills knowledge using 2,500 held-out test samples over 5 epochs.
   - **Status**: Scientifically sound, but structurally hardcoded. It currently hardcodes paths to Stage 3B (teacher), Stage 5A (student), and Stage 4 (reference). It must be parameterized to support Phase B.

3. **FedEraser (`run_federaser.py`)**:
   - **Mechanism**: Replays the federated trajectory round-by-round, skipping the target client. For each retained client, it computes a new calibrated update by doing 2 local epochs on the current reconstructed global model, and uses it to update the global model.
   - **Status**: Scientifically sound, but structurally hardcoded to Stage 3B and Stage 4. Furthermore, it explicitly loads a JSON file of evaluation indices from a specific past KD run to ensure fair comparisons. It must be parameterized.

## B. SCIENTIFIC VALIDITY / CONFOUNDS

1. **The Matched Counterfactual ($M_{CF}$)**:
   - *Verification*: The implementation in `run_federated.py` successfully utilizes `drop_clients_post_sampling`. This guarantees that the underlying `torch.randperm` state remains perfectly synchronized with the Attack conditions. The 19 honest clients undergo the exact same batch sampling, learning sequence, and aggregation participation across Early/Middle/Late and $M_{CF}$. 
   - *Conclusion*: $M_{CF}$ is the scientifically perfect control for client-level unlearning in this experiment.
2. **Confound - Mismatched Baselines**: 
   - If the unlearning scripts evaluate fidelity against the old Stage 4 model (which diverged via RNG resampling), the fidelity metrics will be artificially depressed. Evaluation *must* target $M_{CF}$.
3. **Confound - Proxy Data Variability**:
   - KD uses 2,500 samples as a proxy dataset. If this subset changes between Early, Middle, and Late, the KD efficiency might fluctuate. Fortunately, the subset generation uses a fixed `torch.Generator().manual_seed(42)`, ensuring identical proxy data across all conditions.

## C. PHASE B-1 EXPERIMENT MATRIX

**Goal**: Benchmark baseline unlearning methods against temporal influence placement.
- **Conditions**: Early, Middle, Late.
- **Unlearning Methods**:
  1. Historical Subtraction
  2. Subtraction + KD
  3. FedEraser
- **Target Reference Model**: $M_{CF}$

This requires exactly 9 unlearning executions.

## D. REQUIRED CODE CHANGES (Minimal)

To execute Phase B scientifically without hardcoding, the following minimal code modifications are necessary:

1. **`scripts/unlearn_stage5b.py` (KD)**:
   - *Change*: Add `argparse` for `--teacher_path`, `--student_path`, `--reference_path`, and `--out_dir`.
   - *Why*: Allows the script to distill the Early model into the Early Subtraction model, evaluated against $M_{CF}$.

2. **`scripts/run_federaser.py` (FedEraser)**:
   - *Change*: Add `argparse` for `--source_dir`, `--reference_path`, and `--out_dir`. Replace the hardcoded JSON loading of `eval_indices` with the same deterministic generation logic used in `unlearn_stage5b.py`.
   - *Why*: Decouples the algorithm from Stage 3B/4, enabling it to reconstruct Early/Middle/Late paths and evaluate fidelity against $M_{CF}$ directly.

## E. METRIC + EVALUATION PROTOCOL

**Primary Metrics**:
1. **Counterfactual Fidelity (Prediction Agreement with $M_{CF}$)**: This is the most crucial metric. Since earlier influence might alter the trajectory permanently, measuring how closely an unlearning algorithm recovers the un-attacked $M_{CF}$ trajectory determines if the method is "trajectory-aware."
2. **Final ASR**: Measures backdoor removal efficacy. A method that achieves 0% ASR but has low fidelity is merely "forgetting", not "unlearning."

**Secondary Metrics**:
1. **Clean Accuracy**: Must remain close to $M_{CF}$ baseline.
2. **Compute Cost (Time/Epochs)** & **Communication Cost**: Demonstrates the efficiency trade-off of FedEraser vs. Subtraction.

**Trajectory Analysis**:
While the final ASR is the benchmark metric, observing the *Phase A training trajectory* (e.g., ASR peaking and decaying in the Early condition) will be critical during the final analysis to explain *why* methods succeed or fail.

## F. DRY-RUN PLAN

To validate the pipeline without burning 80 minutes of compute:
1. Implement the `argparse` refactoring.
2. Run Subtraction on **Early** condition.
3. Run KD on **Early** condition (uses the Subtraction output).
4. Run FedEraser on **Early** condition using `--smoke-test` (reconstructs only 2 rounds instead of 100).
5. Verify outputs, metrics, and ensure no crashes.

## G. FULL-RUN PLAN

After dry-run validation:
1. Write a `run_stage6d_phase_b.py` orchestrator script.
2. Sequentially execute Subtraction -> KD -> FedEraser for Early.
3. Repeat for Middle.
4. Repeat for Late.
5. Aggregate all metrics into a final benchmark report.
6. Assess whether the counterfactual gap varies systematically with temporal placement.

## H. GO / NO-GO DECISION

**GO**. 
The Phase A outputs are scientifically pristine. The matched counterfactual solves the Stage 4 trajectory confound. The existing baseline algorithms are fundamentally sound and only require parameterization. We are clear to proceed with the Dry Run.
