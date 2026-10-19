# Baseline Results Summary

This page gives a concise, partition-aware summary. Exact tabular values are in the CSVs under results/.

## Phase 1: clean vs attacked

| Partition | Model | Clean accuracy | ASR/TTR | UTR |
|---|---|---:|---:|---:|
| IID | M_clean | 75.11% | 2.73% | 3.18% |
| IID | M_attack | 75.01% | 56.07% | 3.41% |
| Non-IID | M_clean | 70.10% | 3.30% | 3.84% |
| Non-IID | M_attack | 69.80% | 32.21% | 4.00% |

## Phase 2: principal endpoints

### IID

| Model | Clean accuracy | ASR/TTR | Macro-F1 |
|---|---:|---:|---:|
| M_attack | 75.01% | 56.07% | ~74.8% |
| M_sub | 66.88% | 0.24% | 66.76% |
| M_KD | 72.82% | 2.40% | 72.73% |
| FedEraser Ecal=1 | 72.11% | 2.67% | 71.84% |
| FedEraser Ecal=2 | 74.99% | 2.34% | 74.86% |
| M_CF reference | 74.87% | 3.0111% | 74.70% |

### Non-IID — one saved partition / seed

| Model | Clean accuracy | ASR/TTR | Macro-F1 |
|---|---:|---:|---:|
| M_attack | 69.80% | 32.21% | 68.69% |
| M_sub | 28.04% | 0.87% | 17.10% |
| M_KD | 60.36% | 3.69% | 57.67% |
| FedEraser Ecal=1 | 65.75% | 3.53% | 64.21% |
| FedEraser Ecal=2 | 68.89% | 3.09% | 67.60% |
| M_CF reference | 69.64% | 3.40% | 68.42% |

## Takeaways that are supported by these records

1. Clean accuracy barely changes between clean and attacked training, while ASR/TTR rises sharply.
2. Direct subtraction suppresses TTR but has a severe utility cost, especially in the saved Non-IID run.
3. KD and FedEraser recover much of the lost utility in the recorded runs.
4. FedEraser Ecal=2 is a strong observed balance among tested settings; it is not identical to M_CF on every metric.
5. The current Non-IID evidence is a single saved partition/seed.
