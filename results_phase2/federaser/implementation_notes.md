# FedEraser Implementation Notes

## Algorithm Settings
- **Original Unlearning Paper**: *FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models (Liu et al., 2021)*
- **Retention Interval**: $\Delta t = 2$.
- **Original Global Communication Rounds**: $50$.
- **Retained Rounds Schedule**: 1, 3, 5, 7, ..., 49. (Total 25 reconstructed steps).
- **Client Weighting Policy**: Consistent with Dirichlet distribution remaining participation, we utilize normalized sample weighting for the non-malicious pool: $p_k = n_k / \sum_{j=1}^9 n_j$.

## Training Configuration
- **Dataset**: CIFAR-10
- **Model**: SmallCNN
- **Local Original Epochs**: $E_{loc} = 2$
- **Local Calibration Epochs**: $E_{cali} = 1$
- **Optimizer**: SGD (LR=0.01, Momentum=0.9, Batch Size=64)

## Structural Constraints & Norm Granularity
- **Update Representation**: The historical deltas are taken strictly as parameter-wise PyTorch state dictionaries representing the difference between the local updated model and the preceding global model.
- **First-Step Calibration Bypassed**: At retained round 1, because the starting global model ($G^0$) contains absolute zero influence from C0, client updates are not recalibrated via gradient descent. The native, normalized Round-1 historical updates for C1-C9 are aggregated directly.
- **Norm Granularity**: FedEraser calibrates historical updates via $\tilde{U} = \|U\|_{2} \frac{\hat{U}}{\|\hat{U}\|_{2}}$. In this implementation, the $L_2$ norm is computed globally over the flattened vector of all floating-point parameters (Weights and Biases). 
- **BatchNorm Policy**: Explicitly matching standard verified unlearning logic (`src/unlearning/federaser.py`), non-trainable tracking variables and integer variables (e.g., `running_mean`, `running_var`, `num_batches_tracked`) completely bypass magnitude scaling to prevent precision destruction. They are securely cloned directly from the unpolluted calibration delta.
