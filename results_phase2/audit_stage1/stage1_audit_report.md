# Stage 1: Final Source-of-Truth Audit

## 1. Repository and Frozen State
- **Git Status:** The core repository tracked files are totally unmodified (`git status --short` returned no modifications for tracked files). The `results_phase2/audit_stage1/` directory outputs remain untracked / ignored.
- **Tag Targets:** 
  - `phase1-frozen`: `db5a1c0`
  - `phase2-subtraction-kd-frozen`: `b118294`
- **Git Log Analysis:** The commit history reveals that the repository was initialized in retrospect; the initial commit (`db5a1c0`) serves as a comprehensive snapshot of the Phase-1 codebase and artifacts at the point version control was activated, rather than a contemporaneous historical commit of the original run.
- **Git Remote:** None (`git remote -v` is empty).

## 2. Canonical Results Table
*Reference: `canonical_results.md` and `canonical_results.csv`*
The standard metrics across Phase 1 and Phase 2 have been extracted directly from the canonical `.json` evaluations. `M_CF`'s Counterfactual Agreement is listed as **Reference**, as self-agreement is tautological. 
- **Macro F1 Normalization:** The Phase-1 output `clean_macro_f1` was normalized dynamically to a common percentage format (e.g., `74.91%`) in the canonical outputs, maintaining finite validity.
- **Data Integrity:** All numeric metrics in `canonical_results.csv` (excluding specific categorical `N/A`s or `Reference` string flags) are structurally validated as finite and correctly scaled. Status: `PASS`.
- **Runtime Tracking:** Runtimes have been structurally segregated into `Runtime_Type` and full-precision numeric `Runtime_Seconds`.

## 3. Non-IID M_sub Collapse
*Reference: `noniid_m_sub_per_class.csv`, `noniid_m_sub_predicted_distribution.csv`, and `.png` plots.*
Evaluating on the 10,000-image test set, the frozen Non-IID $M_{sub}$ checkpoint exhibits a broad reduction in classification capacity across multiple categories:
- Classes 3, 4, 7, and 9 yielded **0.0% clean accuracy**.
- The predicted-class distribution confirms the model assigned **zero predictions** to those four classes. The total valid predictions perfectly sum to `10,000`.

## 4. ASR-Versus-Round Trajectory
*Reference: `asr_vs_round.csv`, `asr_vs_round.png`*
Both IID and Non-IID logged trajectories were successfully loaded directly from the CSV and confirmed to contain exactly 50 consecutive rounds of strictly finite values. 
- **IID ASR Late-stage verification**: The ASR at Round 42 was `38.39%` and at Round 50 was `56.07%`. This corresponds strictly to an **increase of 17.68 percentage points** over this late-stage interval.

## 5. Storage Inventory Profile
*Reference: `artifact_storage_inventory.csv`*
The full storage inventory confirms massive buffer overhead:
- `results_phase2/iid_attack_logged/trajectory`: `1.1 GB`
- `results_phase2/noniid_attack_logged/trajectory`: `1.1 GB`

**Runtime Note:** Comparing the "Full Training" standard FL runtime against the "Logged Training" runtime to calculate logging overhead is officially **inconclusive** given the uncontrolled/asynchronous hardware scopes in which Phase-1 and Phase-2 were historically executed. 

## 6. M_CF Aggregation Implementation
Source: `scripts/run_noniid_m_cf.py` & `src/training/fedavg.py`
The code physically isolates C0 from the Dirichlet averaging pool:
```python
        # OMIT CLIENT 0
        for client_id in range(1, config["num_clients"]):
            ...
            sample_counts.append(len(partitions[client_id]))
        # FedAvg naturally renormalizes via sum(sample_counts)
        global_state = aggregate_fedavg(global_state, client_deltas, sample_counts)
```
The exact sizes correctly sum dynamically in `aggregate_fedavg` to denominator `46,403`.

## 7. Historical Evaluations and Proxy Data
- **Evaluation Consistency:** *Reference: `historical_eval_diff.txt`*. Maximum absolute difference strictly $< 1e-4$. Status: `FINAL PASS`.
- **Proxy Provenance:** 
  - Teacher output counts natively sum exactly to 5,000 for IID and Non-IID.
  - Sourced from HF `uoft-cs/cifar100`.
  - Converted natively to a torchvision-compatible pickle format.
  - Required a patched integrity check and a placeholder `meta` file to bypass the standard MD5 verifications securely.
  - The 5,000-image tensor SHA-256 resolves identically to: `630dc3286c8dd16a74a8ac4d6c7ac92fa59dc6868d5a6c6a77b793585f79d4c7`.
  - True labels were utilized strictly for class-balanced sampling and explicitly ignored during the KD distillation loop.

## 8. Git & Backup Audit
- **Git Tracking:** Evaluated Phase-2 logic securely retained on `phase2-subtraction-kd-frozen`. The current local working-tree modifications only consist of the untracked/ignored audit directories.
- **Excluded Files:** `.gitignore` totally isolates `trajectory/` buffers. 
- **Backup Verification:** No active git remote. Given this, backup validation for the `trajectory` files remains **NOT VERIFIED**.

---

## 9. Final Checklist

| Item | Status | Evidence Source |
| :--- | :--- | :--- |
| **Git Status** | **PASS** | Evaluated via standard tracked logs. Modification resides safely in ignored directories. |
| **Canonical Table / Scaled Integrity** | **PASS** | Validated natively across parsed JSON dictionaries and Python conversion schemas. |
| **M_sub Collapse Validation** | **PASS** | Measured perfectly over complete 10,000 inference bounds. |
| **ASR Trajectory Logs** | **PASS** | Analyzed natively on fixed 50-round sequential boundaries. |
| **Storage Inventory** | **PASS** | Extracted via Python `os.walk` footprint. |
| **M_CF Normalization Logic** | **PASS** | Established algorithmically against baseline index boundaries. |
| **Eval Consistency Verification** | **PASS** | Measured rigorously against absolute $< 1e-4$ tolerances. |
| **Artifact Backups** | **NOT VERIFIED** | No verifiable independent storage replication pipeline active. |
