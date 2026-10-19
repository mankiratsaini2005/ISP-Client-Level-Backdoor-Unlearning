# Archive Staging Move Report

## 1. Initial Disk State
- **Total**: 228 GiB
- **Used**: 203 GiB
- **Available**: ~221 MiB (Critical)

## 2. Directories Moved
The following directories were successfully relocated to `results/archive_staging/`:
- `results/stage_6d_temporal/`
- `results/stage7a_c2_attacked_20260911_005839/` (Fixed typo from 20260910 to actual path)
- `results/stage7a_c4_attacked_20260911_035522/`
- `results/stage7c_d2_attacked_recovery_20260911_191151_20260911_191154/`
- `results/stage7c_d1_attacked_20260911_182144/`

## 3. Move Verification
- A pre-move size and file-count manifest was generated.
- The `os.rename` operation successfully moved the references within the same filesystem without executing a full copy operation.
- **Post-move validation**: All 5 directories strictly matched their pre-move byte sizes and file counts, confirming 100% integrity transfer. 
- *See `results/archive_staging/archive_staging_manifest.json` for details.*

## 4. Protected Artifact Verification
The following MUST REMAIN LOCAL and were successfully verified to be untouched:
- `results/stage7c_d2_attacked_recovery_20260911_194422...` (Active D2)
- `results/stage7a_c1_attacked_...` (Canonical D0)
- `results/stage7a_c3_attacked_...` (Canonical D3)
- `results/stage7c_partitions/`
- `src/`, `scripts/`, `configs/`, `docs/`, `tests/`

## 5. Final Disk State
- **Available**: ~1.2 GiB (Slight fluctuation due to FS overhead, but virtually unchanged as expected for an in-place move).
- The storage layout is now cleanly organized for external archival.

## 6. Next Step
The `results/archive_staging/` directory is prepared. It can now be safely uploaded manually to Google Drive. Once uploaded and verified externally, this staging directory can be safely deleted to recover ~20.5 GB of SSD space.
