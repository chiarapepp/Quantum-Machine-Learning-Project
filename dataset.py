import pandas as pd

# Load the Parquet file
df = pd.read_parquet("NF_UNSW_NB15/NF-UNSW-NB15.parquet")

# Display info
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
