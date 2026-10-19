# FedEraser Seed Manifest

**Versions**:
- Python: 3.11.6
- PyTorch: 2.14.0

**Seeding Function** (from `scripts/run_federaser_v5.py`):
```python
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
```

**Exact Seed Expression**:
```python
seed_val = hash((42, round_num, c)) % (2**32)
seed_everything(seed_val)
g = torch.Generator()
g.manual_seed(seed_val)
```

**Scope**:
This determinism lock is applied immediately before instantiating the DataLoader for every single local client epoch across all 50 retained rounds, for 9 retained clients (`c`). It applies uniformly to `iid_Ecal1_v2`, `iid_Ecal2_v2`, `noniid_Ecal1_v2`, and `noniid_Ecal2_v2`.
