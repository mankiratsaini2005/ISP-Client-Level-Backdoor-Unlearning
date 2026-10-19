# Removing Backdoors from Federated Models via Machine Unlearning

## Project Purpose
This repository houses the research code for studying the effects of machine unlearning on backdoor attacks in Federated Learning (FL). 
The project aims to systematically compare:
1. Clean Federated Learning
2. Backdoor/Poisoning Attack on FL
3. Identifying malicious clients
4. Machine Unlearning (e.g., FedEraser)
5. Comparison with Full Retraining

**Current Milestone:** Centralized Baseline.

> [!WARNING]
> **Future Milestones:** Federated Learning, backdoor attacks, data partitioning across clients, and unlearning are **NOT** implemented yet. Currently, this repository only contains the centralized baseline for CIFAR-10.

## Environment Setup

The project requires Python 3.x and is managed using a local virtual environment.
We have tested on macOS with Apple Silicon (MPS supported). CUDA is also supported if available.

### Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Primary Dependencies:* `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`, `pytest`.

## Project Structure

```
project_root/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── configs/                  # Experiment configurations (.yaml)
├── src/                      # Source code
│   ├── data/                 # Datasets and dataloaders
│   ├── models/               # Model architectures
│   ├── training/             # Training and evaluation loops
│   └── utils/                # Utilities (seed, logging, viz)
├── scripts/                  # Executable scripts (e.g., run_centralized.py)
├── tests/                    # Automated tests
├── checkpoints/              # Saved model checkpoints (ignored by git)
├── results/                  # Experiment results and figures (ignored by git)
└── logs/                     # General logs (ignored by git)
```

## Running the Centralized Baseline

To execute the centralized training on CIFAR-10:

```bash
source .venv/bin/activate
python scripts/run_centralized.py
```

### Configuration
The baseline uses `configs/centralized_baseline.yaml`. You can modify this file to adjust hyperparameters without altering the source code.
Configurable parameters include:
- `seed`: Controls randomness (default `42`)
- `batch_size`: Default `64`
- `epochs`: Default `10`
- `learning_rate`: Default `0.01`
- `momentum`: Default `0.9`
- `weight_decay`: Default `5e-4`
- `device_preference`: Default `"mps"` (will fallback to `cuda` or `cpu` automatically).

### Outputs
Upon running the baseline, a new directory under `results/run_YYYYMMDD_HHMMSS/` is created containing:
- `config.json`: A copy of the exact configuration used.
- `metrics.csv`: Epoch-level training and testing metrics (Loss, Accuracy, Time).
- `checkpoints/`: Contains `latest.pt` and `best.pt`.
- `figures/`: Contains automatically generated plots for Loss vs Epoch and Accuracy vs Epoch.
