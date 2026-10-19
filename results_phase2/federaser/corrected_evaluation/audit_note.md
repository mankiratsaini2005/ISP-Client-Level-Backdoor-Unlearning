# FedEraser Corrected Evaluation Audit

## 1. Metric Definitions & Denominators
- **Clean Accuracy/Loss**: Evaluated strictly across all **10,000** test images from the full CIFAR-10 test set.
- **ASR/TTR**: Evaluated over exactly **9,000** non-Airplane test images with the bottom-right trigger applied, targeting Class 0 (Airplane).
- **UTR (Untriggered Target Rate)**: Evaluated over the **9,000** untriggered non-Airplane test images, isolating the count of false predictions for Class 0.
- **Airplane Recall**: Evaluated strictly against the **1,000** true Airplane (Class 0) test images.
- **Clean CF Agreement**: Computed as the agreement percentage between the unlearned model and $M_{CF}$ across all **10,000** unpolluted test images.
- **Triggered CF Agreement**: Computed as the agreement percentage between the unlearned model and $M_{CF}$ across the **9,000** triggered non-Airplane test images.
- **Macro F1**: Computed analytically as the strict macro-averaged $F_1$ score (average of per-class harmonic mean of precision and recall) over the full **10,000** image predictions.

## 2. Partition Equality
The dynamic validation script (`scripts/verify_partitions.py`) confirmed that the client partitions dynamically generated via the predetermined seed completely equaled the logged configurations:
- **IID Exact Equality:** TRUE
- **Non-IID Exact Equality:** TRUE

## 3. Configuration Findings
- **Weight Decay Deviation:** Investigation into Phase-1 records (`run_experiment_v2.py`) revealed that `weight_decay=5e-4` was **NOT** used during original Phase-1 training (the baseline strictly utilized `SGD(lr=0.01, momentum=0.9)`). The presence of `weight_decay` in the existing `scripts/run_baseline_federaser.py` was inherited from a deprecated archived script. (This is documented; no retraining was run in this task).
- **Calibration Shuffling:** In `run_baseline_federaser.py`, `DataLoader(partitions[c], shuffle=True)` is initialized without an explicitly anchored per-round generator seed. This introduces mild stochasticity into batch formation during calibration updates.

## 4. Algorithm Implementation Details
- **Update Sign Convention**: $\Delta = W_{cali} - G_{recon}$, meaning updates are aggregated via $G_{new} = G_{recon} + \Delta$.
- **Norm Granularity**: Computed entirely as a singular global $L_2$ norm tracking over all mathematically flattened floating-point tensors collectively.
- **Zero-Norm Handling**: Defended analytically (`scale_factor = hist_norm / cali_norm` bypasses strictly if `cali_norm > 1e-8`).
- **BatchNorm Buffer Policy**: Integer metrics (`num_batches_tracked`) and running statistics bypass magnitude scaling logic strictly and are assigned via standard clone duplication, mirroring the implementation within `src/unlearning/federaser.py`.
- **Remaining-Client Weights**: Consistently preserved as $n_k / \sum_{j=1}^9 n_j$. 

## 5. Runtimes
- **IID Calibration/Reconstruction Runtime**: 326.38 seconds (representing the sum of 25 post-attack calibration rounds).
- **Non-IID Calibration/Reconstruction Runtime**: 336.01 seconds.
- **Requirement Constraint**: The massive initial training duration (2,600 - 4,200s) and subsequent $1.1\text{ GB}$ storage caching latency natively required to execute offline FedEraser pipelines natively precede this unlearning phase.

## 6. Limitations
Because the existing FedEraser checkpoints were frozen directly by this task mandate, the evaluations generated above structurally inherit the undocumented `weight_decay=5e-4` divergence inside the calibration epochs. No alterations were performed on the underlying network parameters.
