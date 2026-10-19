import torch
import os
from src.models.small_cnn import SmallCNN

model = SmallCNN()
state_dict = model.state_dict()

total_elements = 0
for tensor in state_dict.values():
    total_elements += tensor.numel()

# float32 is 4 bytes
bytes_per_model = total_elements * 4
mb_per_model = bytes_per_model / (1024 * 1024)

# 10 clients * 50 rounds * 1 model
# 50 global rounds * 1 model
total_models = (10 * 50) + 50

total_mb = mb_per_model * total_models

print(f"Elements per model: {total_elements}")
print(f"Size per state_dict (MB): {mb_per_model:.4f} MB")
print(f"Total Trajectory Models (10 clients * 50 rounds + 50 global): {total_models}")
print(f"Estimated Trajectory Storage: {total_mb:.2f} MB")
