import pandas as pd
from sklearn.utils import shuffle
import numpy as np

#shuffling of datasets done here
df = pd.read_csv("./LC_loans_granting_model_dataset.csv")


df = shuffle(df, random_state=42).reset_index(drop=True)
n_parts = 10
subsets = np.array_split(df, n_parts)
for i, subset in enumerate(subsets):
    subset.to_csv(f"lendingclub_partition_{i+1}.csv", index=False)

print(f"Created {n_parts} partitions")
