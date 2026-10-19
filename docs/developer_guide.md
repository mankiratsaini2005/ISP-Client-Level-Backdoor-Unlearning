# ISP Project: Teammate Developer & Reproducibility Guide

Welcome to the **Client-Level Backdoor Unlearning in Federated Learning** repository. This guide provides actionable instructions for cloning, setting up the environment, running tests, executing unlearning baselines, and maintaining experimental integrity.

---

## 1. Repository Setup & Dependencies

### Clone the Repository
```bash
git clone https://github.com/mankiratsaini2005/ISP-Client-Level-Backdoor-Unlearning.git
cd ISP-Client-Level-Backdoor-Unlearning
git checkout docs/team-handoff
```

### Environment Installation
The project dependencies are specified in `requirements.txt`:
```text
torch
torchvision
numpy
pandas
matplotlib
pyyaml
tqdm
pytest
```
- **Tested & Verified Environment**: Python 3.11.6, PyTorch 2.14.0, Torchvision 0.29.0, NumPy 2.4.6, Pandas 3.0.5, Matplotlib 3.11.1, PyYAML 6.0.3, tqdm 4.70.0, pytest 9.1.1.
- **Compatibility Notice**: All experiments, numerical baselines, and test suites were verified exclusively under Python 3.11.6 and PyTorch 2.14.0. While standard `torch.nn`, `torch.optim`, and `weights_only=True` support are used, compatibility with other Python versions (such as Python 3.10 or 3.12+) or other PyTorch versions is explicitly **unverified**. Teammates should strictly use the verified environment to ensure deterministic reproducibility.

Create and activate the virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment:
```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__, '| CUDA/MPS:', torch.cuda.is_available() or hasattr(torch.backends, 'mps'))"
```

---

## 2. Canonical Experiment Protocol & Configurations

The canonical setup evaluates client-level backdoor unlearning on CIFAR-10 with a SmallCNN architecture under two partition regimes:
1. **IID Partition**: 10 clients, uniform random partitioning (`seed=42`).
2. **Non-IID Partition**: 10 clients, Dirichlet distribution ($\alpha=0.5$, `seed=42`).

### Core Protocol Parameters
- **Rounds**: 50 rounds total.
- **Clients**: 10 total clients (C0–C9). Client 0 is the malicious target client to be unlearned.
- **Participation**: Full participation (all 10 clients participate in every round).
- **Target Backdoor**: 
  - Malicious client: Client 0 (C0).
  - Poisoning rate: **10%** of C0's local dataset (~500 samples in IID; ~359 samples in Non-IID Dirichlet $\alpha=0.5$).
  - Trigger: $3 \times 3$ white patch placed in the bottom-right corner of the image (`trigger_value = (1.0 - mean) / std`).
  - Target class: Class 0 (Airplane).
  - Attack activation: **Rounds 30–50 inclusive** (Rounds 1–29 are pre-attack clean federated training).
- **Local Optimizer**: SGD, learning rate $\eta = 0.01$, momentum $\mu = 0.9$, batch size 64, explicit `weight_decay = 0.0`.
- **Local Epochs**: $E = 2$ local training epochs per round.

### Canonical Run Configurations
Configuration files are stored in `configs/`:
- `configs/v2_iid.yaml`: IID federated training specification.
- `configs/v2_noniid.yaml`: Non-IID federated training specification.
- `configs/v2_smoke.yaml`: Fast 2-round smoke-test configuration for validation.

---

## 3. Running Smoke Tests & Verification

### Active Unit Test Suite
The repository includes an active test suite covering core components (data partitioning, backdoor trigger injection, SmallCNN architecture, FedAvg delta aggregation, deterministic seeding, and FedEraser update calibration):
```bash
# Runs all 15 active unit tests (100% pass)
pytest
```
*Result: 15 passed in ~1.4s (100% green).*

### Historical Prototype Tests
Tests written for earlier exploratory prototype stages (100-round September 9 runs and Stage 7b runner) are preserved under `tests/historical/` for complete auditability:
```bash
# Run historical prototype tests explicitly
pytest tests/historical/
```
*Result: 6 failed in ~2.5s (expected).*
- **5 failures in `tests/historical/test_benchmark_freeze.py`** (`test_canonical_directories_exist`, `test_canonical_checkpoints_load`, `test_evaluation_splits_frozen`, `test_target_client_and_participation_history_frozen`, `test_data_partition_frozen`): These fail because they assert on superseded prototype directories from September 9 exploratory runs (`results/attack_scaling_final_gamma5_20260909_015903`, etc.) that were superseded by the Phase 1 & Phase 2 canonical benchmarks.
- **1 failure in `tests/historical/test_stage7b_counterfactual.py`** (`test_counterfactual_exclusion`): Fails with `FileNotFoundError: 'configs/federated_baseline.yaml'` because it relies on an early prototype configuration and Stage 7b runner.
*(These tests are preserved for historical provenance without weakening assertions, but are isolated from the active suite via `pytest.ini`).*

### Fast End-to-End Smoke Test
Run a lightweight 2-round CIFAR-10 FL cycle:
```bash
python test_cifar.py
```

---

## 4. Experiment Rerunnability & Trajectory Prerequisites

> [!IMPORTANT]
> **What can run immediately vs. what requires trajectory regeneration**:
> The Git repository tracks all evaluated global model checkpoints, configurations, logs, and evaluation scripts (~75 MB total). However, to avoid exceeding Git storage limits, large raw per-round client delta directories (`trajectory/`, ~2.2 GB) and pickled datasets (`client_partitions.pt`, ~1.48 GB) are excluded.

### A. What Runs Immediately (Without Retraining)
Using the committed checkpoints in `results/` and `results_phase2/`, the following commands execute immediately:
- **Extended Evaluation**:
  ```bash
  # Evaluate Phase-1 Clean models against M_CF
  PYTHONPATH=. python scripts/eval_p1_cf.py

  # Evaluate Non-IID Phase-1 clean models
  PYTHONPATH=. python scripts/eval_noniid_clean.py

  # Evaluate committed FedEraser checkpoints
  PYTHONPATH=. python scripts/eval_only_federaser.py

  # Evaluate Phase-2 unlearning endpoints
  PYTHONPATH=. python scripts/evaluate_phase2_corrected.py
  ```
- **Summary Table & Metric Compilation**:
  ```bash
  # Recompile canonical comparison table from evaluation JSONs
  PYTHONPATH=. python scripts/compile_v3_results.py

  # Verify BatchNorm tracking counters across checkpoints
  PYTHONPATH=. python scripts/check_batches_tracked.py

  # Audit run-to-run metric differences
  PYTHONPATH=. python scripts/diff_runs.py
  ```
- **Plotting & Visualizations**:
  ```bash
  PYTHONPATH=. python scripts/plot_results.py
  PYTHONPATH=. python scripts/per_class_asr.py
  ```

### B. What Requires Trajectory Regeneration
FedEraser (`scripts/run_federaser_v5.py`) and historical update subtraction (`scripts/run_subtraction.py`) require the following inputs:
- `results_phase2/{iid,noniid}_attack_logged/trajectory/round_{r:03d}/delta_client_{c:02d}.pt` (Rounds 1–50, Clients 1–9)
- `results_phase2/{iid,noniid}_attack_logged/client_partitions.pt`
- `results_phase2/{iid,noniid}_attack_logged/initial_model.pt` *(already committed)*
- `results_phase2/{iid,noniid}_counterfactual/final_model.pt` *(already committed)*

To regenerate these missing trajectory deltas and partition files from scratch, run the logged attack training script:
```bash
# 1. Regenerate IID logged attack trajectory (50 rounds):
PYTHONPATH=. python scripts/run_experiment_v2.py \
    --config configs/v2_iid.yaml \
    --mode attack \
    --log_trajectory \
    --output_dir results_phase2/iid_attack_logged

# 2. Regenerate Non-IID logged attack trajectory (50 rounds):
PYTHONPATH=. python scripts/run_experiment_v2.py \
    --config configs/v2_noniid.yaml \
    --mode attack \
    --log_trajectory \
    --output_dir results_phase2/noniid_attack_logged
```

> [!NOTE]
> **Trajectory Output Counts & Validation Status**:
> - **File Counts**: Each trajectory generation command executes 50 rounds with 10 clients per round, generating exactly **500 client-update delta files per setting** (`trajectory/round_{r:03d}/delta_client_{c:02d}.pt` for rounds 1–50 and clients 0–9), or **1,000 client-update files combined** across both IID and Non-IID if all updates are successfully logged. Each round also saves the round-level `global_model.pt` checkpoint, alongside `client_partitions.pt` and `client_class_histograms.json`.
> - **Validation Status**: The full 50-round trajectory regeneration has **not** been executed end-to-end to avoid long retraining times and to strictly protect existing baseline artifacts; full 50-round regeneration remains **unverified end-to-end**. However, the script mechanics, directory creation, per-client delta serialization, and partition metadata generation were verified via a controlled 1-round smoke test with temporary outputs. Note that running trajectory regeneration requires CIFAR-10 data extracted at `./data/cifar-10-batches-py`.


Once regenerated, full FedEraser reconstruction and subtraction can be executed:
```bash
# Run FedEraser baselines (E_cal = 1 and E_cal = 2):
PYTHONPATH=. python scripts/run_federaser_v5.py

# Run Historical Subtraction:
PYTHONPATH=. python scripts/run_subtraction.py
PYTHONPATH=. python scripts/run_noniid_subtraction.py

# Run Knowledge Distillation Recovery:
PYTHONPATH=. python scripts/run_kd.py
```

### C. Retraining Counterfactual Reference ($M_{\text{CF}}$)
To retrain the independent counterfactual reference model without Client 0 from scratch (does not require trajectory deltas):
```bash
# IID M_CF:
PYTHONPATH=. python scripts/run_m_cf.py

# Non-IID M_CF:
PYTHONPATH=. python scripts/run_noniid_m_cf.py
```

---

## 5. Artifact Locations & Evaluation Tables

- **Phase-1 Baseline Models**: `results/iid_clean/`, `results/iid_attack/`, `results/noniid_clean/`, `results/noniid_attack/`.
- **Phase-2 Unlearning Models**: `results_phase2/iid_subtraction/`, `results_phase2/iid_kd/`, `results_phase2/iid_counterfactual/`, and Non-IID equivalents.
- **FedEraser Checkpoints & Logs**: `results_phase2/federaser/` (see `RUN_REGISTRY.md` and `corrected_evaluation_v2/federaser_comparison.md`).
- **Seed Manifest**: `results_phase2/federaser/seed_manifest.md`.

---

## 6. How to Add a New Run Without Overwriting
1. Create a dedicated output subdirectory under `results_phase2/` or `results_phase2/federaser/` (e.g., `results_phase2/federaser/iid_Ecal3/`).
2. Record the exact configuration, seeds, and optimizer hyperparameters in `config.json` inside the run directory.
3. Update `results_phase2/federaser/RUN_REGISTRY.md` with your new run's rationale and status.
4. Run the extended evaluation protocol (`evaluate_extended`) to generate `eval_results.json`.
5. Recompile the summary table using `scripts/compile_v3_results.py`.

---

## 7. Known Reproducibility Notes
- **Python `hash()` Seed Caveat**: In `scripts/run_federaser_v5.py`, seed derivation uses `hash((42, round_num, client_id)) % (2**32)`. Note that Python's built-in `hash()` is randomized across distinct Python interpreter sessions unless the environment variable `PYTHONHASHSEED` is set to a fixed integer (e.g., `export PYTHONHASHSEED=42`).
- **BatchNorm Tracking Buffers**: In SmallCNN, `features.1.num_batches_tracked` and `features.5.num_batches_tracked` are integer buffers initialized to 0. During FedEraser aggregation and historical subtraction, integer tracking buffers must bypass scaling and remain strictly unchanged.
- **Historical Timing Comparability**: Historical $M_{\text{CF}}$ wall-clock measurements included 3 synchronous evaluation passes per round across 28,000 examples plus final checkpoint saving. Do not compare raw $M_{\text{CF}}$ wall-clock time directly against local compute microbenchmarks.
