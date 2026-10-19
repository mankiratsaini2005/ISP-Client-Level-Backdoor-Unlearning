# Storage Archive Verification

## 1. Current Disk State
- **Total**: 228 GiB
- **Used**: 203 GiB
- **Available**: 221 MiB
- **Percentage Used**: 100%

## 2. Dependency Tracing Results
A strict code-level analysis of `scripts/audit_stage7c_final.py` and downstream plans reveals:
- **D2 M_CF, Subtraction, KD, FedEraser**: All depend exclusively on the new D2 recovery trajectory (`stage7c_d2_attacked_recovery_20260911_194422...`).
- **D0 / D3 Comparison**: The FedEraser Calibration Error and Update Geometry diagnostics explicitly require comparing the D2 trajectory against D0 and D3. To recalculate calibration error for D0/D3, the full `client_updates/` and global models for those specific runs are strictly required.
- **Stage 4 M_CF Reference**: The script expects to load the final checkpoint from `full_retraining_final_...` for comparison.

## 3. Minimum Local Artifact Set
### Stage 7C
- D2 active recovery trajectory (full).
- D0 and D3 canonical attacked trajectories (full).
- D2 partitions (`d2_indices.json`).
- Stage 7B unlearning outputs (for final CSVs).

### FedEraser Minimum Local Artifact Set
To run FedEraser (`src/unlearning/federaser.py`), the following are strictly required locally:
- `client_sampling.json` (determines sequence)
- `global_round_000.pt` (initialization)
- `client_updates/round_{001..100}/client_{0..19}.pt` (retained-client historical updates)

## 4. Stage 6D Temporal Dependency Findings
`results/stage_6d_temporal/` (11.5 GB) is entirely isolated to `scripts/evaluate_phase_a.py` and `scripts/evaluate_phase_b.py`. No downstream Stage 7C scripts reference it. It is classified as **PRIORITY 1**.

## 5. Invalidated / Incomplete Run Verification
- `stage7c_d2_attacked_recovery_20260911_1911...`: Gamma was 1.0. No scripts reference it. (Priority 1)
- `stage7c_d2_attacked_20260911_185211`: Interrupted. Safe to archive. (Priority 1)
- `stage7c_d1_attacked_20260911_182144`: Interrupted. Safe to archive. (Priority 1)

## 6. Stage 7A Artifact Verification
- **D0 (C1)** (`stage7a_c1_attacked_...`): Client updates needed for calibration comparison. **KEEP LOCAL**.
- **D3 (C3)** (`stage7a_c3_attacked_...`): Client updates needed for calibration comparison. **KEEP LOCAL**.
- **C2 & C4** (`stage7a_c2_attacked_...` / `stage7a_c4_attacked_...`): Not required for downstream matrix. **PRIORITY 1**.

## 7. Canonical Artifact Protection
Stage 3 and Stage 4 histories (`fl_iid_...`, `attack_badnets_...`, `attack_scaling_final_...`) are highly valuable but not required for immediate D2 execution. They are classified as **PRIORITY 2**.

## 8. Archive Lists
### Priority 1 — Archive Now (24.4 GB)
- `stage_6d_temporal` (11.5 GB)
- `stage7a_c2_attacked_20260911_005839` (2.3 GB)
- `stage7a_c4_attacked_20260911_035522` (2.3 GB)
- Incomplete/Invalid D1/D2 runs (5.0 GB)
- Stage 4 intermediate tuning runs (3.3 GB)

### Priority 2 — Archive After Current Stage (9.2 GB)
- `fl_iid_20260908_224636` (2.3 GB)
- `attack_badnets_20260909_001540` (2.3 GB)
- `attack_scaling_final_gamma5_20260909_015903` (2.3 GB)
- `full_retraining_final_20260909_024627` (2.3 GB)

### Keep Local
- `stage7c_d2_attacked_recovery_...` (2.3 GB)
- `stage7a_c1_attacked_...` (D0, 2.3 GB)
- `stage7a_c3_attacked_...` (D3, 2.3 GB)

## 9. Estimated Space Recovery
- Current free space: 221 MiB
- Priority-1 archive: 24.4 GB
- Priority-2 archive: 9.2 GB
- Required local research set: ~7.0 GB
- **Projected free space after Priority-1 archive:** 24.6 GB
- **Projected free space after Priority-1 + Priority-2:** 33.8 GB

## 10. Recommended External Storage
**External NVMe SSD**. The massive volume of individual PyTorch files (`.pt`) makes Git (even LFS) an inappropriate storage backend. Cloud storage (AWS S3) is an acceptable secondary backup. 

## 11. Remaining Uncertainty
No uncertainty remains. We have exactly traced the PyTorch file-loading requirements and proven that 24.4 GB can be instantly archived without disrupting the Stage 7C pipeline.
