import os
import csv
import random
import pandas as pd

# Config
csv_input = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/test/S06/c041/dataset.csv"
train_csv = "train.csv"
val_csv = "val.csv"
split_ratio = 0.8  # 80% train, 20% val
random.seed(42)

# Load dataset
df = pd.read_csv(csv_input)

# Unique vehicle IDs
ids = list(df["id"].unique())
random.shuffle(ids)

# Split by IDs
split_idx = int(len(ids) * split_ratio)
train_ids = set(ids[:split_idx])
val_ids = set(ids[split_idx:])

# Filter rows
train_df = df[df["id"].isin(train_ids)]
val_df = df[df["id"].isin(val_ids)]

# Save CSVs
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

print(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
print(f"CSV files saved: {train_csv}, {val_csv}")
