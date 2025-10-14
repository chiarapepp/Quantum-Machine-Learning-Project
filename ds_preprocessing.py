# dataset_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_parquet("NF_UNSW_NB15/NF-UNSW-NB15.parquet")

print("Original shape:", df.shape)

# Step 1 — Keep only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Ensure we keep the Label column for later
if 'Label' not in numeric_df.columns:
    numeric_df['Label'] = df['Label']

print("Numeric shape:", numeric_df.shape)

# Step 2 — Separate features and labels
X = numeric_df.drop(columns=['Label'])
y = numeric_df['Label']

# Step 3 — Min–Max normalization to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 4 — Quantization into 1440 bins
num_bins = 1440
X_quantized = np.floor(X_scaled * num_bins) / num_bins

# Step 5 — Angle encoding (map [0,1] → [0,π])
X_angles = X_quantized * np.pi

print("✅ Preprocessing done!")
print("Features shape:", X_angles.shape)
print("Labels shape:", y.shape)
print("Example encoded feature row:\n", X_angles[0])
