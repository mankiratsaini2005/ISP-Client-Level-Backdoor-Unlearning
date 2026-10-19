# ISP Project: Client-Level Backdoor Unlearning in Federated Learning

Research workspace for the ISP project **Removing Backdoors from Federated Models via Machine Unlearning**.

The core objective is to evaluate whether the influence of a single malicious federated client (Client 0) can be removed to eliminate a backdoor while preserving utility on clean tasks. The repository implements and compares:
1. **Direct Historical-Update Subtraction ($M_{\text{sub}}$)**
2. **Subtraction followed by Knowledge Distillation ($M_{\text{KD}}$)**
3. **FedEraser Reconstruction ($M_{\text{FedEraser}}$ with $E_{\text{cal}} \in \{1, 2\}$)**
4. **Counterfactual Reference Retraining ($M_{\text{CF}}$)**: Independent reference trained from scratch on clean clients (C1–C9).

---

## 1. Quick Start & Setup

### Environment Setup
The project requires Python 3.10+ and PyTorch $\ge 2.0.0$ (tested baseline: Python 3.11.6, PyTorch 2.14.0).
```bash
# Clone the repository and checkout the working branch
git clone https://github.com/mankiratsaini2005/ISP-Client-Level-Backdoor-Unlearning.git
cd ISP-Client-Level-Backdoor-Unlearning
git checkout docs/team-handoff

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests & Verification
```bash
# Run the active unit test suite (15/15 passed)
PYTHONPATH=. pytest

# Run a fast 2-round CIFAR-10 FL smoke test
PYTHONPATH=. python test_cifar.py
```

---

## 2. Repository Directory Map

```text
ISP-Client-Level-Backdoor-Unlearning/
├── src/                                     # Core model, data, federated, and unlearning logic
│   ├── data/                                # CIFAR-10 data loaders, backdoor injection, partition logic
│   ├── models/                              # SmallCNN model architecture
│   ├── training/                            # FedAvg aggregation & training loops
│   ├── unlearning/                          # FedEraser calibration (with L2-norm parity assertion)
│   └── utils/                               # Seeding, logging, and visualization utilities
├── scripts/                                 # 114 training, unlearning, evaluation, and audit scripts
│   ├── run_experiment_v2.py                 # Logged FL attack training (generates trajectories)
│   ├── run_m_cf.py & run_noniid_m_cf.py     # Counterfactual reference retraining (M_CF)
│   ├── run_subtraction.py                   # Historical update subtraction (M_sub)
│   ├── run_kd.py                            # Knowledge distillation unlearning (M_KD)
│   ├── run_federaser_v5.py                  # Canonical FedEraser baselines (E_cal in {1, 2})
│   ├── eval_p1_cf.py & eval_noniid_clean.py # CF agreement and extended evaluation scripts
│   └── _archive/                            # Archived historical scripts and prototype runners
├── configs/                                 # Canonical experiment YAML specifications
│   ├── v2_iid.yaml                          # IID 50-round FL specification
│   ├── v2_noniid.yaml                       # Non-IID Dirichlet (alpha=0.5) specification
│   ├── v2_smoke.yaml                        # Fast smoke-test configuration
│   └── _archive/                            # Archived run configurations
├── tests/                                   # Active unit test suite
│   ├── historical/                          # Archived prototype tests (isolated from active suite)
│   └── [test_*.py]                          # Core unit tests (attack, data, model, federaser, etc.)
├── docs/                                    # Detailed documentation, guides, and audit logs
│   ├── developer_guide.md                   # Complete teammate developer & reproducibility guide
│   ├── EXPERIMENT_PROTOCOL.md               # Canonical 50-round experiment specifications
│   ├── METHODS_AND_METRICS.md               # Mathematical formulations and metric definitions
│   └── [analysis reports]                   # Factorial robustness audits and stage reports
├── results/                                 # Phase-1 baseline models, metrics, CSVs, and figures
│   ├── iid_clean/ & iid_attack/             # Model checkpoints, per-round metrics, client histograms
│   ├── noniid_clean/ & noniid_attack/       # Model checkpoints, per-round metrics, client histograms
│   └── [CSVs & PNGs]                        # Baseline metric snapshots, heatmaps, and curves
├── results_phase2/                          # Phase-2 unlearning models, evaluations, and artifacts
│   ├── federaser/                           # Corrected v2 runs, RUN_REGISTRY.md, seed_manifest.md
│   ├── iid_subtraction/ & noniid_subtraction/ # M_sub checkpoints and timing metadata
│   ├── iid_kd/ & noniid_kd/                 # M_KD checkpoints and metadata
│   ├── iid_counterfactual/ & noniid_counterfactual/ # M_CF checkpoints and round metrics
│   ├── audit_stage1/                        # Canonical audit logs and reports
│   └── proxy_data/                          # CIFAR-100 proxy dataset indices and sample grid
├── presentation/                            # Editable PowerPoint slide decks
│   ├── current/                             # ClientLevel_Backdoor_Unlearning_FL_initial_draft.pptx
│   └── historical/                          # Removing_Backdoors_from_Federated_Models.pptx and decks
└── manifests/                               # Provenance and checksum manifests
    └── source_import_manifest.md            # Detailed record of imported files and exclusions
```

---

## 3. Experiment Rerunnability & Trajectory Prerequisites

> [!IMPORTANT]
> **Reproduction Limitation & Trajectory Requirements**:
> All evaluated global model checkpoints (`final_model.pt`, `initial_model.pt`), configurations, evaluation metrics, and audit scripts are committed to Git (~75 MB total).
>
> However, to comply with GitHub storage constraints, the large per-round client delta directories (`trajectory/`, ~2.2 GB) and pickled dataset objects (`client_partitions.pt`, ~1.48 GB) are excluded from the repository.

### Commands That Run Immediately (No Retraining Required)
Using the committed checkpoints in `results/` and `results_phase2/`:
- **Model Evaluation**:
  ```bash
  PYTHONPATH=. python scripts/eval_only_federaser.py
  PYTHONPATH=. python scripts/eval_p1_cf.py
  PYTHONPATH=. python scripts/eval_noniid_clean.py
  PYTHONPATH=. python scripts/evaluate_phase2_corrected.py
  ```
- **Result Summarization & Audit**:
  ```bash
  PYTHONPATH=. python scripts/compile_v3_results.py
  PYTHONPATH=. python scripts/check_batches_tracked.py
  PYTHONPATH=. python scripts/diff_runs.py
  ```
- **Plotting**:
  ```bash
  PYTHONPATH=. python scripts/plot_results.py
  PYTHONPATH=. python scripts/per_class_asr.py
  ```

### Commands Requiring Trajectory Regeneration
FedEraser (`scripts/run_federaser_v5.py`) and historical subtraction (`scripts/run_subtraction.py`) require historical per-round client updates. To regenerate them:
```bash
# Regenerate IID logged attack trajectory (50 rounds):
PYTHONPATH=. python scripts/run_experiment_v2.py \
    --config configs/v2_iid.yaml \
    --is_attack \
    --log_trajectory \
    --output_dir results_phase2/iid_attack_logged

# Regenerate Non-IID logged attack trajectory (50 rounds):
PYTHONPATH=. python scripts/run_experiment_v2.py \
    --config configs/v2_noniid.yaml \
    --is_attack \
    --log_trajectory \
    --output_dir results_phase2/noniid_attack_logged
```
This produces all 500 per-round client updates (`trajectory/round_XXX/delta_client_YY.pt`) and `client_partitions.pt`, enabling subsequent execution of `scripts/run_federaser_v5.py` and `scripts/run_subtraction.py`.

---

## 4. Key Experimental Baseline Observations

- **Clean accuracy alone does not detect the backdoor**: In Phase 1 IID, clean accuracy was virtually identical (75.11% clean vs. 75.01% attack), while the Attack Success Rate (ASR) jumped from 2.73% to 56.07%.
- **Direct Subtraction ($M_{\text{sub}}$)**: Eliminates the backdoor but causes catastrophic utility collapse under Non-IID partitions (28.04% clean accuracy in the saved split).
- **Knowledge Distillation Recovery ($M_{\text{KD}}$)**: Partially recovers clean accuracy (72.82% IID, 60.36% Non-IID) with low ASR.
- **FedEraser Reconstruction**: Reaches 74.99% accuracy / 2.34% ASR in IID ($E_{\text{cal}}=2$) and 68.89% accuracy / 3.09% ASR in the saved Non-IID partition ($E_{\text{cal}}=2$).
- **Counterfactual Model ($M_{\text{CF}}$)**: Serves as an empirical reference point trained without Client 0 (74.87% IID, 69.64% Non-IID).

For complete protocol details, consult [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) and [docs/developer_guide.md](docs/developer_guide.md).
