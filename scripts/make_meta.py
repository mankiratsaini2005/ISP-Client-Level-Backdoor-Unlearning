import pickle
import os

os.makedirs("data/cifar-100-python", exist_ok=True)
with open("data/cifar-100-python/meta", "wb") as f:
    pickle.dump({"fine_label_names": [f"class_{i}" for i in range(100)]}, f)
