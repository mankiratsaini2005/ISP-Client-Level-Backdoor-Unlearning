# Stage 7A Pre-Flight Report

## 1. Frozen C1-C4 Matrix
- **C1**: H0 (IID) + A0 (Canonical Attack, $\gamma=5$)
- **C2**: H1 ($\alpha=0.5$) + A0 (Canonical Attack, $\gamma=5$)
- **C3**: H2 ($\alpha=0.1$) + A0 (Canonical Attack, $\gamma=5$)
- **C4**: H2 ($\alpha=0.1$) + A1 (Stronger Scaling Stress, $\gamma=10$)

## 2. Full H0/H1/H2 Partition Statistics (Benign Clients 1-19)
**H0 (IID)**:
- Total Benign Samples: 47,500
- Min/Max Samples: 2,500 / 2,500 (Mean: 2500.0, Std: 0.0)
- Min/Max Class Entropy: 2.301 / 2.301 (Mean: 2.301)
- Min/Max Classes/Client: 10 / 10

**H1 (Moderate Non-IID)**:
- Total Benign Samples: 47,500
- Min/Max Samples: 836 / 5,455 (Mean: 2500.0, Std: 1050.8)
- Min/Max Class Entropy: 1.206 / 2.062 (Mean: 1.714)
- Min/Max Classes/Client: 8 / 10

**H2 (Strong Non-IID)**:
- Total Benign Samples: 47,500
- Min/Max Samples: 116 / 12,266 (Mean: 2500.0, Std: 2849.0)
- Min/Max Class Entropy: 0.467 / 1.473 (Mean: 0.909)
- Min/Max Classes/Client: 4 / 10

## 3. Client-0 Statistics
- **Total Samples**: 2,500
- **Class Distribution**: `{0:254, 1:237, 2:248, 3:240, 4:256, 5:244, 6:265, 7:261, 8:260, 9:235}`
- **Class Entropy**: 2.301
- Client 0 is completely identical across H0, H1, and H2. The heterogeneity manipulation is purely applied to the benign-client population while the malicious client's local clean dataset is held fixed.

## 4. H2 Computational-Workability Check
- **Smallest H2 client size**: 116 samples.
- **FL Configuration**: `batch_size=64`, `drop_last=False`.
- **Validation**: This client produces exactly 2 batches per local epoch (sizes 64 and 52). No empty batches occur. No batch has a size of 1, which strictly prevents any BatchNorm tensor shape crashes. The FL system will process this client stably.

## 5. A0/A1 Configuration
- **A0 (Canonical)**: `gamma=5.0`, `attack_start=30`, `attack_end=100`. (Identical to Stage 3B).
- **A1 (Stronger Stress)**: `gamma=10.0`, `attack_start=30`, `attack_end=100`. (Differs from A0 *only* in update scaling magnitude).

## 6. Control Reuse
TOTAL UNIQUE TRAINING RUNS = 10.
- 4 attacked runs (C1, C2, C3, C4)
- 3 unique No-Attack controls (C1, C2, C3)
- 3 unique matched counterfactuals (C1, C2, C3)
C4 is $H2 + A1$. It shares the exact same benign heterogeneous data environment (H2) and the exact same malicious target client (Client 0) as C3 ($H2 + A0$). Therefore, C4 strictly and legitimately reuses C3's No-Attack control and C3's matched counterfactual, saving 2 full federated training runs.

## 7. Counterfactual Definitions
The Matched Counterfactual ($M_{CF}$) for H0/H1/H2 is explicitly defined as a fresh, separate training run using the identical initialization, partition seed, sampling schedule, and optimizer as the attacked condition. Client 0 is stochastically sampled identically, but its updates are deterministically dropped before aggregation, without drawing a replacement client. Stage 4 is not used.

## 8. Artifact Retention Plan
- **Attacked Runs (4x)**: Retain `best_global.pt`, `global_round_000.pt`, and the full 100-round `client_updates` archive. (Mandatory for historical FedEraser replay).
- **Controls & $M_{CF}$ (6x)**: Retain only `best_global.pt` for final evaluation. The system will skip saving `client_updates` entirely (`--skip_client_updates`).

## 9. Storage Calculation & 10. Current Disk Space
- **Current Free Disk**: 12.0 GB
- **Estimated Attacked-Run Footprint (4x)**: 9.2 GB
- **Estimated Control / M_CF Footprint (6x)**: 0.024 GB
- **Estimated Unlearning-Output Footprint (12x)**: 0.05 GB
- **Total Expected Stage-7A Footprint**: ~9.3 GB
- **Expected Remaining Free Disk**: ~2.7 GB (Estimated Safe)

## 11. Code-Change Audit
- The `scripts/stage7a_generate_partitions.py` isolates controlled heterogeneity without touching canonical logic.
- Storage flags (`--skip_client_updates`, `--save_final_only`) must be safely implemented in `run_federated.py` via argument parsing before Phase B execution, defaulting to standard behavior to preserve canonical stability.

## 12. Validation Results
Automated validation (`scripts/validate_stage7a.py`) passed all 16 checks seamlessly, confirming that H0/H1/H2 are reproducible, heterogeneity strictly increases, Client-0 is fixed, controls are safely shared, and canonical Stage 3B parameters are perfectly aligned.

## 13. Remaining Threats to Validity
- **Non-IID Convergence Failure**: Strong Non-IID settings ($\alpha=0.1$) often cause federated non-convergence. If the C3 No-Attack control fails to reach usable utility, C3's attack condition might fail the Attack-Validity Gate.
- **Update Geometry Saturation**: The stronger stress condition (`gamma=10.0`) might push the local updates out of standard FP16/FP32 precision bands depending on aggregation scales, requiring gradient clipping checks.

STAGE 7A PRE-FLIGHT PASSED — READY FOR TRAINING
