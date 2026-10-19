# Storage Optimization Audit

## 1. Current Disk State
- **Total Disk**: 228 GiB
- **Used**: 203 GiB
- **Available**: 260 MiB
- **Percentage Used**: 100% capacity. Space is critically low.

## 2. Project-Level Storage Breakdown
| Directory | Size | % of Project |
|---|---:|---:|
| `results/` | 41 GB | 96.9% |
| `.venv/` | 931 MB | 2.2% |
| `data/` | 340 MB | 0.8% |
| `scripts/` | 308 KB | < 0.1% |
| `tests/` | 156 KB | < 0.1% |
| `docs/` | 124 KB | < 0.1% |
| `figures/` | 120 KB | < 0.1% |
| `src/` | 92 KB | < 0.1% |
| `configs/` | 20 KB | < 0.1% |

## 3. Top 10 Largest Storage Consumers
The primary storage consumers are the `client_updates/*.pt` histories. Each full FL trajectory (100 rounds * 10 clients/round) generates approximately 1,000 `.pt` files, consuming roughly 2.3 GB per run.
1. `results/stage_6d_temporal/` (11.5 GB, 3 sub-trajectories)
2. `results/stage7c_d2_attacked_recovery_20260911_194422_20260911_194423` (~2.3 GB)
3. `results/stage7c_d2_attacked_recovery_20260911_191151_20260911_191154` (~2.3 GB, Invalidated)
4. `results/stage7a_c4_attacked_20260911_035522` (~2.3 GB)
5. `results/stage7a_c3_attacked_20260911_022619` (~2.3 GB, D3 Canonical)
6. `results/stage7a_c2_attacked_20260911_005839` (~2.3 GB)
7. `results/stage7a_c1_attacked_20260910_233012` (~2.3 GB, D0 Canonical)
8. `results/attack_scaling_final_gamma5_20260909_015903` (~2.3 GB)
9. `results/full_retraining_final_20260909_024627` (~2.3 GB)
10. `results/fl_iid_20260908_224636` (~2.3 GB)

## 4. Stage 7C Dependency Audit
The immediate downstream D2 tasks (M_CF, Subtraction, KD, FedEraser, and Diagnostics) explicitly require:
| Artifact | Required for | Required locally? | Can be archived? |
|---|---|---|---|
| Active D2 Recovery Run | All downstream unlearning & geometry | YES | NO |
| D2 M_CF (To be generated) | All evaluations | YES | NO |
| D0 (C1) Attacked Run | Final calibration & geometry comparison | YES | NO |
| D3 (C3) Attacked Run | Final calibration & geometry comparison | YES | NO |

## 5. FedEraser Dependency Audit
By analyzing `scripts/run_federaser.py`, the algorithm requires:
1. Complete `client_updates/` directory representing the historical trajectory.
2. Complete `checkpoints/` directory for historical global norms and base weights.
3. `client_sampling.json` for participant tracking.
4. `client_partitions.json` to calculate retained-client dataset sizes.
If any of these are archived, FedEraser cannot reconstruct the trajectory.

## 6. Invalidated Experiments
- `stage7c_d2_attacked_recovery_20260911_191151_20260911_191154`: Gamma was incorrectly set to 1.0 (Invalidated).
- `stage7c_d2_attacked_20260911_185211`: Interrupted at round 29 (Incomplete).
- `stage7c_d1_attacked_20260911_182144`: Interrupted at round 29 (Incomplete).
*These are safe to archive externally, recovering ~5.1 GB.*

## 7. Proposed Storage Tiers
**TIER 1 — LOCAL ACTIVE (~9.2 GB)**
- Active D2 recovery run
- D0 (C1) Canonical run
- D3 (C3) Canonical run
- D1 and D2 partition data

**TIER 2 — LOCAL CANONICAL (~0.5 GB)**
- Exact unlearning output checkpionts (Subtraction, KD)
- Stage 7B Final Audit models
- Final metrics/CSV files

**TIER 3 — EXTERNAL ARCHIVE (~31.3 GB)**
- `stage_6d_temporal/` (11.5 GB)
- `stage7a_c2_attacked_...` (2.3 GB)
- `stage7a_c4_attacked_...` (2.3 GB)
- `attack_scaling_final_gamma5_...` (2.3 GB)
- `full_retraining_final_...` (2.3 GB)
- `fl_iid_...` (2.3 GB)
- `attack_badnets_...` (2.3 GB)
- All Invalidated/Incomplete runs (5.1 GB)

## 8. Final Recommendation
1. **Current Disk State:** 100% full, 260 MiB available. Action is critically needed before training D2 M_CF.
2. **Top 10 consumers:** Detailed above; heavily dominated by `client_updates/` trajectories.
3. **Artifacts required for D2 downstream:** Active D2 recovery run, D0 canonical run, D3 canonical run, and partition definitions.
4. **Artifacts to archive:** Tier 3 directories (listed above). 
5. **Estimated Space Recovered:** ~31.3 GB.
6. **Recommended External Storage Method:** Cloud object storage (e.g., AWS S3, GCS) or an External NVMe SSD. Ordinary Git is *highly inappropriate* for thousands of `.pt` files.
7. **NEVER DELETE:** Stage 7C partitions, Stage 7B/7C audit metrics, D0/D3/active-D2 client updates, or historical documentation (`docs/`).
8. **Remaining uncertainty:** None. We have a clear deterministic split of what FedEraser requires locally vs what is safe to archive.

STORAGE AUDIT COMPLETE — NO FILES MODIFIED
WAITING FOR USER AUTHORIZATION BEFORE ANY ARCHIVE/DELETE ACTION
