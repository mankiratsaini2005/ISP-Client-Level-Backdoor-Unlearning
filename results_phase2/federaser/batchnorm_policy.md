# FedEraser BatchNorm Buffer Policy (Pre-Training Documentation)

Before running the evaluation of the 4 specified FedEraser configurations ($\Delta t=1$, $E_{cal} \in \{1, 2\}$), the following explicit BatchNorm buffer policy is documented to avoid any silent algorithmic substitutions:

## Floating-Point Trainable Parameters (Weights/Biases)
All standard floating-point variables that do not include `running_` or `num_batches_tracked` in their name are explicitly subjected to FedEraser's $L_2$-norm magnitude scaling:
$\tilde{U}_i = ||U_i||_2 \frac{\hat{U}_i}{||\hat{U}_i||_2}$
The norm $|| \cdot ||_2$ is computed as a singular global scalar aggregating all such tensors.

## BatchNorm Running Statistics and Counters
Non-trainable metrics specifically designated for statistical tracking (i.e. `running_mean`, `running_var`) and integer variables (e.g., `num_batches_tracked`) strictly **bypass** the global L2 magnitude calibration. 

Instead, the baseline implementation directly clones these buffers from the newly obtained calibration update $\hat{U}_i$ without subjecting them to a fractional scale factor. This preserves their intrinsic tracking scale from the unpolluted local epochs. 

No weighted averaging, recalculation, or separate global recalculation pass is applied to these terms beyond FedAvg's standard data-weighted aggregation across the selected non-malicious clients ($C_1 - C_9$). This preserves the exact algorithmic mechanics without introducing undocumented variant repairs.
