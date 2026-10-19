import json
import numpy as np
from itertools import combinations

with open("results/stage7c_partitions/d2_indices.json", "r") as f:
    indices = json.load(f)

with open("results/stage7c_partitions/d2_distribution.json", "r") as f:
    dist = json.load(f)

sizes = [len(idx) for idx in indices.values()]
print(f"Sizes: {sizes}")
print(f"Min: {np.min(sizes)}")
print(f"Max: {np.max(sizes)}")
print(f"Mean: {np.mean(sizes)}")
print(f"Std: {np.std(sizes)}")
print(f"CV: {np.std(sizes)/np.mean(sizes)}")

def gini(array):
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

print(f"Gini: {gini(np.array(sizes))}")

def entropy(p, q=None):
    if q is None:
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    else:
        idx = (p > 0) & (q > 0)
        return np.sum(p[idx] * np.log2(p[idx] / q[idx]))

entropies = []
dists = []
for k, v in dist.items():
    cd = v["class_distribution"]
    arr = [cd.get(str(i), 0) for i in range(10)]
    arr = np.array(arr, dtype=float)
    arr /= arr.sum()
    dists.append(arr)
    entropies.append(entropy(arr))
    
print(f"Mean entropy: {np.mean(entropies)}")

def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

js_divs = []
for p, q in combinations(dists, 2):
    js_divs.append(js_div(p, q))
    
print(f"Mean JS Div: {np.mean(js_divs)}")
