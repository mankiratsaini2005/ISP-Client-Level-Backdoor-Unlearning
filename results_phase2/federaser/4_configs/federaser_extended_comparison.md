| Partition | Model | Clean Acc | Clean Loss | ASR/TTR | UTR | Clean CF Agree | Triggered CF Agree | Air. Recall | Macro F1 | Runtime Type | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IID | Phase-1 Clean | 75.11% | 0.7148 | 2.73% | 3.18% | 93.01% | 92.77% | 80.50% | 74.91% | Full Training | 2754.3257 |
| IID | Attack (No unlearning) | 75.01% | 0.7188 | 56.07% | 3.41% | 92.77% | 41.88% | 82.00% | 74.80% | Logged Training | 2664.2736 |
| IID | M_sub | 66.88% | 0.9354 | 0.24% | 0.56% | 76.25% | 77.90% | 40.60% | 66.76% | Subtraction-only | 0.1772 |
| IID | M_KD | 72.82% | 0.8192 | 2.40% | 3.94% | 88.54% | 86.79% | 81.40% | 72.73% | Subtraction+KD | 7.8164 |
| IID | M_FedEraser_Ecal1 | 72.43% | 0.8017 | 3.18% | 3.63% | 86.08% | 85.68% | 81.90% | 72.26% | Calibration/Reconstruction | 657.0723 |
| IID | M_FedEraser_Ecal2 | 74.96% | 0.7209 | 2.18% | 2.51% | 90.57% | 90.83% | 77.50% | 74.84% | Calibration/Reconstruction | 1284.7171 |
| IID | M_CF | 74.87% | 0.7179 | 3.01% | 3.62% | Reference | Reference | 82.40% | 74.70% | Retraining | 2513.8698 |
| Non-IID | Phase-1 Clean | 70.10% | 0.8240 | 3.30% | 3.84% | N/A | N/A | N/A | 69.01% | Full Training | 2805.1042 |
| Non-IID | Attack (No unlearning) | 69.80% | 0.8303 | 32.21% | 4.00% | 90.28% | 63.36% | 81.30% | 68.69% | Logged Training | 4266.3362 |
| Non-IID | M_sub | 28.04% | 5.6408 | 0.87% | 1.02% | 35.31% | 36.88% | 14.20% | 17.10% | Subtraction-only | 3.7689 |
| Non-IID | M_KD | 60.36% | 1.0678 | 3.69% | 4.66% | 78.65% | 76.80% | 75.60% | 57.67% | Subtraction+KD | 11.2864 |
| Non-IID | M_FedEraser_Ecal1 | 65.40% | 0.9577 | 4.12% | 5.08% | 84.57% | 83.77% | 81.70% | 63.74% | Calibration/Reconstruction | 673.6084 |
| Non-IID | M_FedEraser_Ecal2 | 69.00% | 0.8499 | 2.54% | 3.14% | 87.92% | 86.94% | 77.90% | 67.89% | Calibration/Reconstruction | 1327.4624 |
| Non-IID | M_CF | 69.64% | 0.8392 | 3.40% | 3.74% | Reference | Reference | 79.00% | 68.42% | Retraining | 4219.1692 |
