import pickle

with open("data/cifar-100-python/train", "rb") as f:
    d = pickle.load(f)

with open("data/cifar-100-python/train", "wb") as f:
    pickle.dump({
        "data": d[b"data"],
        "fine_labels": d[b"fine_labels"]
    }, f)
