import numpy as np
from data import load_dexnet_per_class


counts_per_class = load_dexnet_per_class()
uniques, counts = np.unique(counts_per_class, return_counts=True)
print(uniques, sorted(counts))
