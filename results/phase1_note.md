# Phase 1 Note

During the initiation of the Non-IID experiments in Phase 1, a temporary logic bug was encountered in `src/data/fl_partition.py`. 

Specifically, the line:
`proportions = np.array([p * (len(class_idx) < (total_size / num_clients)) for p in proportions])`
incorrectly evaluated `5000 < 5000` to `False` for the CIFAR-10 training set. This zeroed out all Dirichlet proportions, causing a `ValueError: num_samples=0` when `torch.utils.data.DataLoader` attempted to draw from an empty dataset.

This faulty scalar-clipping line was immediately removed, as `np.random.dirichlet` natively scales to 1.0. 

Because the two IID experiments had already cleanly completed before the bug was encountered, they were preserved. The Non-IID Clean and Non-IID Attack experiments were restarted using the corrected partition logic and completed perfectly.
