# Stage 7C Recovery Audit

## 1. Last Valid Round
The last fully completed round of Federated Learning is **Round 29**. 
The `client_updates/round_029` directory contains a complete set of client updates for that round. However, due to the `--save_final_only` flag used during execution, **no global checkpoint was saved for Round 29**. The latest available global model state is stored in `best_global.pt`, which corresponds to **Round 28** (the round with the highest test accuracy before termination).

## 2. Existing Artifacts
- **Checkpoints**: `global_round_000.pt`, `best_global.pt`
- **Client Updates**: Complete for rounds 000 through 029. Incomplete for round 030 (8 out of 10 client updates present).
- **Metadata**: `config.yaml`, `client_partitions.json`, `client_sampling.json`
- **Logs**: `stage7c_d2_attacked.log`

## 3. Checkpoint Integrity
Loading `best_global.pt` confirms that the file is not corrupted. Tensor shapes and BatchNorm buffers are structurally valid. However, the checkpoint strictly contains the model's `state_dict()` and **lacks all training state**.

## 4. Optimizer and RNG State Availability
- **Optimizer State**: UNAVAILABLE. The checkpoint does not contain the momentum buffers or state of the local SGD optimizers.
- **RNG State**: UNAVAILABLE. The Python, NumPy, and PyTorch RNG states were not saved. 

## 5. Sampling Determinism
Although the client participation schedule is perfectly recorded in `client_sampling.json`, the internal mini-batch shuffling within each client's `DataLoader` relies on the PyTorch RNG state. Without the RNG state, the exact order of data presentation during local training cannot be reproduced, leading to trajectory divergence.

## 6. Partition Integrity
The D2 partition metadata is safely intact in `client_partitions.json` and perfectly matches the target D2 (Label-Balanced / Size-Imbalanced) specification.

## 7. Termination Diagnosis
The run was explicitly terminated by the system infrastructure (`Tool execution was canceled`). The log abruptly terminates at round 29 without any Python exceptions, tracebacks, or out-of-memory errors. The termination was external and deliberate (e.g., a timeout or manual system kill).

## 8. Resume Feasibility
The run is **NOT SAFELY RESUMABLE**. 
Trajectory-based unlearning methods like FedEraser depend heavily on the exact sequence of historical updates. Because the optimizer momentum buffers and RNG state are lost, restarting the run from `best_global.pt` would cause the new updates (rounds 29-100) to diverge from the hypothetical uninterrupted trajectory. This would introduce an uncontrolled discontinuity in the update geometry, invalidating the FedEraser calibration analysis.

## 9. Recommended Recovery Procedure
The D2 attacked run must be fully restarted from Round 0. The incomplete artifacts should be removed or overwritten by a fresh uninterrupted execution to ensure mathematical continuity of the entire 100-round trajectory.
