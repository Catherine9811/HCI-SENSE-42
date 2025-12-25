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

# --- Load participant metadata CSV ---
metadata_file = os.path.join("..", "..", "data", "participant_enrollment.csv")  # <-- update path if needed
meta_df = pd.read_csv(metadata_file)


# --- Clean column names ---
def clean_column_name(col):
    return (
        col
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(".", "")
        .replace(" ", "_")
        .replace(":", "")
        .replace("?", "")
        .replace("-", "_")
        .lower()
    )


meta_df.columns = [clean_column_name(c) for c in meta_df.columns]

# --- Convert Participant ID to integer ---
if "participant_id" not in meta_df.columns:
    raise ValueError("Metadata CSV must contain 'Participant ID' column")

meta_df["participant_id"] = meta_df["participant_id"].astype("int64")

# --- Process 'select ALL that apply' columns ---
select_cols = [c for c in meta_df.columns if "select_all_that_apply" in c or c.startswith("psqi_5")]

for col in select_cols:
    meta_df[col] = (
        meta_df[col]
        .fillna("")
        .apply(
            lambda x: 0 if len(x.strip()) == 0 else x.count(";") + 1
        )
    )

# --- Drop excluded columns ---
exclude_columns = [
    "timestamp",
    "allow_video_published_in_anonymized_form",
    "allow_video_published_in_raw_form",
    "allow_use_for_commercial_purposes",
    # "ess_total"
]

meta_df.drop(
    columns=[c for c in exclude_columns if c in meta_df.columns],
    inplace=True,
)

# --- Merge metadata into main dataframe ---
merged_df = pd.merge(
    merged_df,
    meta_df,
    left_on="participant",
    right_on="participant_id",
    how="left",
)

# Remove duplicate participant_id column after merge
merged_df.drop(columns=["participant_id"], inplace=True)

merged_df.sort_values(by=key_columns, inplace=True)
merged_df.to_csv(output_file, index=False)

print(f"\n✅ Merged CSV saved to: {output_file}")
