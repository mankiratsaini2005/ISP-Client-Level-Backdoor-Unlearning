import numpy as np

np.random.seed(42)

# Remaining(c)
# Client 0 has ~250 per class (let's say exactly 250 for now).
# Total remaining per class = 4750.
col_sums = np.full(10, 4750)
row_sums = np.full(19, 2500)

alpha = 0.1
M = np.random.dirichlet(np.repeat(alpha, 10), size=19)

# Sinkhorn-Knopp
for _ in range(1000):
    # Normalize rows
    M = M * (row_sums[:, None] / M.sum(axis=1)[:, None])
    # Normalize cols
    M = M * (col_sums[None, :] / M.sum(axis=0)[None, :])

M_int = np.floor(M).astype(int)

# Distribute remaining
row_diff = row_sums - M_int.sum(axis=1)
col_diff = col_sums - M_int.sum(axis=0)

while np.sum(row_diff) > 0:
    for r in range(19):
        while row_diff[r] > 0:
            # find a column c where col_diff[c] > 0
            valid_cols = np.where(col_diff > 0)[0]
            if len(valid_cols) == 0:
                break
            # Pick the valid column that maximizes the original M proportion (or random)
            c = valid_cols[np.argmax(M[r, valid_cols])]
            M_int[r, c] += 1
            row_diff[r] -= 1
            col_diff[c] -= 1

print(M_int.sum(axis=1))
print(M_int.sum(axis=0))

entropies = []
for r in range(19):
    p = M_int[r] / 2500
    entropies.append(-np.sum(p[p>0] * np.log(p[p>0])))
print(np.mean(entropies))
