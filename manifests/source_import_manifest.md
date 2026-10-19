# Source Import Manifest: ISP Client-Level Backdoor Unlearning

## 1. Provenance & Source Metadata
- **Source Project Root**: `/Users/mankirat/Desktop/ISP`
- **Import Timestamp**: 2026-10-19T13:40:00+05:30 (Updated with post-transfer audit at 2026-10-19T13:50:00+05:30)
- **Source Git Branch**: `main`
- **Source HEAD Commit**: `b118294` (`Phase 2: Unlearning (Historical Subtraction and KD)`)
- **Source Baseline Tags**:
  - `phase1-frozen` (`db5a1c0`)
  - `phase2-subtraction-kd-frozen` (`b118294`)
- **Uncommitted Working-Tree Changes Transferred**:
  - `src/unlearning/federaser.py`: Update calibration with explicit global L2-norm parity assertion (`assert math.isclose(calibrated_norm, hist_norm)`).
  - `results_phase2/charts/airplane_recall.png`: Regenerated figure.
  - `results_phase2/charts/clean_acc.png`: Regenerated figure.
  - Newly generated FedEraser baseline outputs in `results_phase2/federaser/`: `iid_Ecal1_v2/`, `iid_Ecal2_v2/`, `noniid_Ecal1_v2/`, `noniid_Ecal2_v2/`, `RUN_REGISTRY.md`, `seed_manifest.md`, `controlled_timing.json`, and comparison tables.
  - Newly generated Phase-1 clean extended evaluation records in `results/iid_clean/` and `results/noniid_clean/` (`extended_metrics.json`, `cf_agreement.json`).
  - Audit and compilation scripts in `scripts/`.

---

## 2. Reconciled Tracked Inventory (Directly from Git Index)
Total tracked files in repository (excluding `.git`): **363 files**, totaling **75.00 MB** (**78,640,784 bytes**).

### Breakdown by Top-Level Directory
| Directory | Tracked File Count | Payload Size | Purpose & Content |
|---|---|---|---|
| `src/` | 11 | 23.8 KB | Core FL modules (`src/data/fl_partition.py`, `src/data/cifar10.py`, `src/data/backdoor.py`, `src/models/small_cnn.py`, `src/training/fedavg.py`, `src/training/train_utils.py`, `src/unlearning/federaser.py`, `src/utils/logger.py`, `src/utils/seed.py`, `src/utils/viz.py`, `src/README.md`). |
| `scripts/` | 115 | 451.1 KB | 114 training, unlearning, evaluation, and audit scripts + `scripts/README.md` (including canonical `run_experiment_v2.py`, `run_m_cf.py`, `run_subtraction.py`, `run_kd.py`, `run_federaser_v5.py`). |
| `configs/` | 12 | 5.1 KB | 11 YAML configurations (`v2_iid.yaml`, `v2_noniid.yaml`, `v2_smoke.yaml`, and archived variants) + `configs/README.md`. |
| `tests/` | 10 | 22.6 KB | 8 active unit tests (`test_attack.py`, `test_data.py`, `test_federaser.py`, `test_federaser_toy.py`, `test_federated.py`, `test_model.py`, `test_seed.py`) + `historical/` directory (2 archived tests + README). |
| `docs/` | 36 | 177.5 KB | Canonical experiment protocol, methods, team workflow, developer guide, source project README, factorial analysis, and audit reports. |
| `results/` | 52 | 25.39 MB | Phase-1 clean/attack models (`final_model.pt`, `initial_model.pt`), metrics CSVs, histograms, charts, and preserved summary CSVs. |
| `results_phase2/` | 116 | 46.45 MB | Phase-2 unlearning checkpoints ($M_{\text{sub}}$, $M_{\text{KD}}$, $M_{\text{CF}}$, FedEraser $E_{\text{cal}} \in \{1, 2\}$ for IID and Non-IID), audit logs, charts, and proxy data indices. |
| `presentation/` | 5 | 2.49 MB | 4 editable PowerPoint presentations (current draft and historical versions) + `presentation/README.md`. |
| `manifests/` | 2 | 6.4 KB | `manifests/README.md` and this `source_import_manifest.md`. |
| Root files | 7 | 15.1 KB | `README.md`, `CONTRIBUTING.md`, `.gitignore`, `pytest.ini`, `requirements.txt`, `test_cifar.py`, `sha256_manifest.txt`. |
| **Total** | **366** | **75.01 MB** | **Complete runnable project snapshot.** |

### Reconciliation of Initial Discrepancy
- **Pre-transfer remote branch (`docs/team-handoff`)**: Contained 19 initial files (documentation, READMEs, CSV result snapshots).
- **Import commit (`ac869ae`)**: Staged 342 files from the local project tree and presentations. The combination totaled 361 tracked files. (The earlier manifest draft estimated 347 prior to precise de-duplication with existing remote READMEs and cache exclusions).
- **Corrected post-transfer commit (`d0b9d5e`)**: Added `pytest.ini` and `tests/historical/README.md`, moved 2 historical tests to `tests/historical/`, bringing the tracked file count to 363 files.
- **Source restoration fix (`fix/restore-src-data`)**: Fixed `.gitignore` root directory anchor rule (`/data/` vs `data/`) and restored the 3 missing `src/data/` implementation files (`backdoor.py`, `cifar10.py`, `fl_partition.py`), bringing the final tracked file count to **366 files**.

---

## 3. Artifacts Intentionally Excluded & Justifications
| Category / Path | Size | Reason for Exclusion | Retrieval / Reproduction Instructions |
|---|---|---|---|
| `.venv/` | ~1.1 GB | Python virtual environment with compiled binaries. | Recreate via `python -m venv .venv && pip install -r requirements.txt`. |
| `__pycache__/`, `*.pyc` | ~15 MB | Python bytecode caches. | Automatically generated on execution. |
| `data/` (`cifar-10*`, `cifar-100*`) | ~489 MB | Raw dataset archives and extracted batch files. | Downloaded automatically via `torchvision.datasets.CIFAR10(..., download=True)` or `scripts/download_hf_cifar.py`. |
| `results_phase2/*_attack_logged/trajectory/` | ~2.2 GB | 500 per-setting (1,000 combined across IID & Non-IID) per-round raw client delta checkpoints (`delta_client_XX.pt`, 50 rounds × 10 clients). Exceeds normal Git limits; ignored by project `.gitignore`. | Regenerated by running the logged attack training script (`scripts/run_experiment_v2.py --mode attack --log_trajectory`). |
| `client_partitions.pt` files | ~1.48 GB | PyTorch pickled Subset objects that embed the full raw CIFAR-10 image arrays (~211 MB each). Exceeds GitHub 100 MB hard limit. | Deterministically reproducible using `src/data/fl_partition.py` with `seed=42`. Client class distribution metadata is fully preserved in `client_class_histograms.json`. |

---

## 4. Key Executable Scripts Preserved
- **Federated Training**: `scripts/run_experiment_v2.py`, `scripts/run_four_experiments.sh`, `scripts/run_noniid_experiments.sh`.
- **Counterfactual Reference ($M_{\text{CF}}$)**: `scripts/run_m_cf.py`, `scripts/run_noniid_m_cf.py`.
- **Historical Subtraction ($M_{\text{sub}}$)**: `scripts/run_subtraction.py`, `scripts/run_noniid_subtraction.py`.
- **Knowledge Distillation ($M_{\text{KD}}$)**: `scripts/run_kd.py`.
- **FedEraser Baseline**: `scripts/run_federaser_v5.py` (canonical, strict weight decay 0.0, deterministic seeding, BN buffer bypass).
- **Evaluation & Auditing**: `scripts/eval_p1_cf.py`, `scripts/eval_noniid_clean.py`, `scripts/compile_v3_results.py`, `scripts/final_verification.py`.

---

## 5. Verification & Test Outcomes
- **Python Syntax Compilation**: `python -m compileall` executed over `src/`, `scripts/`, `tests/` with **0 errors**.
- **Verified Environment**: Python 3.11.6, PyTorch 2.14.0, Torchvision 0.29.0, NumPy 2.4.6, Pandas 3.0.5, Matplotlib 3.11.1, PyYAML 6.0.3, tqdm 4.70.0, pytest 9.1.1 (all other versions unverified).
- **Active Unit Test Suite (`pytest`)**:
  - `tests/test_attack.py`: **4 passed** (including `test_update_scaling`, repaired to import `compute_model_delta` from `src.training.fedavg`).
  - `tests/test_data.py`: **1 passed** (deterministic IID & Non-IID partitioning).
  - `tests/test_federaser.py`: **3 passed** (delta calculation, norm calibration, and preservation).
  - `tests/test_federaser_toy.py`: **1 passed** (end-to-end toy FL reconstruction).
  - `tests/test_federated.py`: **3 passed** (weighted aggregation and model delta math).
  - `tests/test_model.py`: **2 passed** (SmallCNN dimensions and parameter names).
  - `tests/test_seed.py`: **1 passed** (deterministic seed reproducibility).
  - **Result: 15 passed in 1.41s (100% green).**
- **Historical Test Suite (`pytest tests/historical/`)**:
  - Preserved in `tests/historical/` to avoid polluting the active test suite with tests targeting superseded prototype folders (`results/attack_scaling_final_gamma5_20260909_015903`, etc.).
  - **Result: 6 failed in ~2.5s (expected)**:
    - 5 failures in `tests/historical/test_benchmark_freeze.py` (`test_canonical_directories_exist`, `test_canonical_checkpoints_load`, `test_evaluation_splits_frozen`, `test_target_client_and_participation_history_frozen`, `test_data_partition_frozen`) due to hardcoded paths to superseded September 9 exploratory directories.
    - 1 failure in `tests/historical/test_stage7b_counterfactual.py` (`test_counterfactual_exclusion`) due to `FileNotFoundError: 'configs/federated_baseline.yaml'`, relying on an early prototype configuration and Stage 7b runner.

EOF
