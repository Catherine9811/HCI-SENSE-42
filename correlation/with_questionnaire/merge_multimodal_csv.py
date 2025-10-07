import pandas as pd
import glob
import os

# --- Configuration ---
key = "sleepiness"
input_folder = "processed_data"
input_filename = f"42-{key}.csv"
output_file = os.path.join("processed_data", input_filename.replace(".csv", "-multimodal.csv"))
key_columns = ["participant", "time"]

# --- Load all CSVs ---
csv_files = glob.glob(os.path.join(input_folder, "*", input_filename))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {input_folder}")

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f" - {f}")

# --- Merge all CSVs by participant & time ---
merged_df = None

for file in csv_files:
    df = pd.read_csv(file)

    # Ensure key columns exist
    if not all(k in df.columns for k in key_columns):
        raise ValueError(f"{file} is missing one of the key columns: {key_columns}")

    # Convert key columns to integer
    for col in key_columns:
        df[col] = df[col].astype("int64")

    # Merge or initialize
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on=key_columns, how="outer", suffixes=("", "_dup"))

# --- Save result ---
merged_df.dropna(inplace=True)  # remove any rows with missing values

# --- Check if sleepiness columns match ---
# Drop duplicate column regardless, keeping 'key'
merged_df.drop(columns=[f"{key}_dup"], inplace=True)

merged_df.sort_values(by=key_columns, inplace=True)
merged_df.to_csv(output_file, index=False)

print(f"\n✅ Merged CSV saved to: {output_file}")
