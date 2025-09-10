import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================
# Load Dataset
# ============================
df = pd.read_csv("paysim.csv")

# Drop irrelevant columns
drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode categorical column 'type'
if "type" in df.columns:
    le = LabelEncoder()
    df["type"] = le.fit_transform(df["type"].astype(str))

# Scale numeric features
num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================
# Create Global Test Set (10%)
# ============================
test_frac = 0.1
test_size = int(len(df) * test_frac)

df_test = df.iloc[:test_size]
df_train = df.iloc[test_size:]

df_test.to_csv("global_test.csv", index=False)
df_train.to_csv("global_train.csv", index=False)

print("Global test shape:", df_test.shape)
print("Global train shape:", df_train.shape)

# ============================
# Partition into 10 clients
# ============================
parts = np.array_split(df_train, 10)
for i, part in enumerate(parts):
    part.to_csv(f"client{i+1}.csv", index=False)
    print(f"Saved client{i+1}.csv with {len(part)} rows")
