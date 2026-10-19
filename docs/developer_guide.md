# ISP Project: Teammate Developer & Reproducibility Guide

Welcome to the **Client-Level Backdoor Unlearning in Federated Learning** repository. This guide provides actionable instructions for cloning, setting up the environment, running tests, executing unlearning baselines, and maintaining experimental integrity.

---

## 1. Repository Setup & Environment

### Clone the Repository
```bash
git clone https://github.com/mankiratsaini2005/ISP-Client-Level-Backdoor-Unlearning.git
cd ISP-Client-Level-Backdoor-Unlearning
```

### Environment Installation
The project requires Python 3.11+ and PyTorch 2.14+.

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment:
```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__, '| CUDA/MPS:', torch.cuda.is_available() or hasattr(torch.backends, 'mps'))"
```

---

## 2. Canonical Experiment Protocol & Configurations

The canonical setup evaluates client-level backdoor unlearning on CIFAR-10 with a SmallCNN architecture under two partition settings:
1. **IID Partition**: 10 clients, uniform random partitioning (`seed=42`).
2. **Non-IID Partition**: 10 clients, Dirichlet distribution ($\alpha=0.5$, `seed=42`).

### Core Protocol Parameters
- **Rounds**: 50 rounds total.
- **Clients**: 10 total clients; Client 0 is the malicious target client to be unlearned.
- **Participation**: Full participation (all active clients per round).
- **Target Backdoor**: Bottom-right trigger ($3 \times 3$ pattern) injected into 50% of Client 0's local samples, targeting Class 0 (Airplane).
- **Local Optimizer**: SGD, learning rate $\eta = 0.01$, momentum $\mu = 0.9$, batch size 64, explicit `weight_decay = 0.0`.
- **Local Epochs**: $E = 2$ for standard training.

### Canonical Run Configurations
Configuration files are stored in `configs/`:
- `configs/v2_iid.yaml`: IID federated training specifications.
- `configs/v2_noniid.yaml`: Non-IID federated training specifications.
- `configs/v2_smoke.yaml`: Fast 2-round smoke-test configuration for validation.

---

## 3. Running Smoke Tests & Verification

Before launching full experiments, run the unit test suite:
```bash
PYTHONPATH=. pytest tests/test_federaser.py tests/test_data.py tests/test_model.py tests/test_federated.py
```

Run a fast end-to-end smoke test:
```bash
PYTHONPATH=. python test_cifar.py
```

---

## 4. Launching Baselines & Unlearning Methods

### A. FedEraser Baseline (Canonical v5 Script)
FedEraser reconstructs the global model across retained rounds 1–50 using calibrated client updates with a global L2 norm constraint:
```bash
# Runs IID and Non-IID under E_cal = 1 and E_cal = 2:
PYTHONPATH=. python scripts/run_federaser_v5.py
```
- Outputs are saved to: `results_phase2/federaser/*_v2/`
- Runtime note: FedEraser reads stored client updates from the logged attack trajectory.

### B. Counterfactual Retraining Reference ($M_{\text{CF}}$)
Retrains the federated model from scratch on clean clients (Clients 1–9) with Client 0 omitted:
```bash
# IID M_CF:
PYTHONPATH=. python scripts/run_m_cf.py

# Non-IID M_CF:
PYTHONPATH=. python scripts/run_noniid_m_cf.py
```

### C. Historical Update Subtraction ($M_{\text{sub}}$)
Directly subtracts the accumulated updates of Client 0 from the global model:
```bash
PYTHONPATH=. python scripts/run_subtraction.py
PYTHONPATH=. python scripts/run_noniid_subtraction.py
```

### D. Knowledge Distillation Recovery ($M_{\text{KD}}$)
Applies knowledge distillation using proxy data to recover clean accuracy after subtraction:
```bash
PYTHONPATH=. python scripts/run_kd.py
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
