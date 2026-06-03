"""
Merge all per-modality CSVs for one lag level into a single multimodal CSV.

Usage (standalone):
    python merge_multimodal_csv.py --lag -1
    python merge_multimodal_csv.py --lag 0 --output-dir processed_data/lag_0

Called programmatically by run_all_lags.py via run_merge().
"""

import argparse
import glob
import os

import pandas as pd


KEY       = "sleepiness"
KEY_COLS  = ["participant", "time"]
N_PARTICIPANTS = 42


def _clean_column_name(col: str) -> str:
    return (
        col
        .replace("/", "").replace("(", "").replace(")", "")
        .replace(",", "").replace(".", "").replace(" ", "_")
        .replace(":", "").replace("?", "").replace("-", "_")
        .lower()
    )


def run_merge(
    lag_base: str,
    n: int | None = None,
    leading_window: float | None = None,
    questionnaire_dir: str | None = None,
    metadata_file: str | None = None,
) -> str:
    """Merge all modality CSVs under *lag_base* and return the output path.

    Parameters
    ----------
    lag_base:
        Directory that contains per-modality sub-directories
        (behavioural/, cardiac/, event7to9/, respiratory/, webcam/).
    n:
        Integer lag step; added as ``lag_n`` column.
    leading_window:
        LEADING_WINDOW in seconds; added as ``leading_window_s`` column.
    questionnaire_dir:
        Path to the directory holding ``42-questionnaires.csv``.
        Defaults to ``<lag_base>/../questionnaire``.
    metadata_file:
        Path to participant enrollment CSV.
        Defaults to ``../../../data/participant_enrollment_with_env.csv``.
    """
    input_filename = f"{N_PARTICIPANTS}-{KEY}.csv"

    # ── Collect per-modality CSVs ────────────────────────────────────────────
    csv_files = glob.glob(os.path.join(lag_base, "*", input_filename))
    if not csv_files:
        print(f"  [merge] No CSVs found in {lag_base} — skipping.")
        return ""

    print(f"  [merge] Found {len(csv_files)} modality CSV(s) in {lag_base}")
    merged_df = None
    for path in sorted(csv_files):
        df = pd.read_csv(path)
        if not all(k in df.columns for k in KEY_COLS):
            raise ValueError(f"{path} is missing key columns {KEY_COLS}")
        for col in KEY_COLS:
            df[col] = df[col].astype("int64")
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(
                merged_df, df,
                on=KEY_COLS, how="outer",
                suffixes=("", "_dup"),
            )

    merged_df.dropna(inplace=True)

    # Drop any duplicate outcome columns created by outer join
    dup_cols = [c for c in merged_df.columns if c.endswith("_dup")]
    merged_df.drop(columns=dup_cols, inplace=True)

    # ── Participant metadata ─────────────────────────────────────────────────
    if metadata_file is None:
        metadata_file = os.path.join("..", "..", "..", "data",
                                     "participant_enrollment_with_env.csv")
    if os.path.exists(metadata_file):
        meta = pd.read_csv(metadata_file)
        meta.columns = [_clean_column_name(c) for c in meta.columns]
        if "participant_id" not in meta.columns:
            raise ValueError("Metadata CSV must contain 'Participant ID' column")
        meta["participant_id"] = meta["participant_id"].astype("int64")

        # Process 'select all that apply' multi-value columns
        select_cols = [c for c in meta.columns
                       if "select_all_that_apply" in c or c.startswith("psqi_5")]
        for col in select_cols:
            meta[col] = (
                meta[col].fillna("").apply(
                    lambda x: 0 if len(x.strip()) == 0 else x.count(";") + 1
                )
            )

        meta.drop(columns=[c for c in [
            "timestamp",
            "allow_video_published_in_anonymized_form",
            "allow_video_published_in_raw_form",
            "allow_use_for_commercial_purposes",
        ] if c in meta.columns], inplace=True)

        merged_df = pd.merge(
            merged_df, meta,
            left_on="participant", right_on="participant_id",
            how="left",
        )
        merged_df.drop(columns=["participant_id"], inplace=True, errors="ignore")
    else:
        print(f"  [merge] Metadata file not found at {metadata_file} — skipping.")

    # ── Questionnaire data ───────────────────────────────────────────────────
    if questionnaire_dir is None:
        questionnaire_dir = os.path.join(lag_base, "..", "questionnaire")
    q_file = os.path.join(questionnaire_dir, f"{N_PARTICIPANTS}-questionnaires.csv")

    if os.path.exists(q_file):
        q_df = pd.read_csv(q_file)
        required = ["name", "time", "value", "initiation", "participant"]
        if not all(c in q_df.columns for c in required):
            raise ValueError(f"Questionnaire CSV must contain {required}")
        q_df["time"]        = q_df["time"].astype("int64")
        q_df["participant"] = q_df["participant"].astype("int64")

        outcome_time = (
            q_df[q_df["name"] == KEY][["participant", "initiation", "time"]]
            .drop_duplicates()
        )

        q_pivot = (
            q_df.pivot_table(
                index=["participant", "initiation"],
                columns="name",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )
        q_pivot.columns.name = None
        q_pivot = pd.merge(q_pivot, outcome_time, on=["participant", "initiation"], how="left")

        merged_df = pd.merge(
            merged_df, q_pivot,
            on=["participant", "time", KEY],
            how="left",
        )
        print("  [merge] Questionnaire data merged.")
    else:
        print(f"  [merge] Questionnaire file not found at {q_file} — skipping.")

    # ── Add lag metadata columns ─────────────────────────────────────────────
    if n is not None:
        merged_df["lag_n"] = n
    if leading_window is not None:
        merged_df["leading_window_s"] = leading_window

    merged_df.sort_values(by=KEY_COLS, inplace=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = os.path.join(lag_base, f"{N_PARTICIPANTS}-{KEY}-multimodal.csv")
    merged_df.to_csv(out, index=False)
    print(f"  [merge] Saved {len(merged_df)} rows → {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lag", type=int, default=0,
                    help="Lag step N (also sets LEADING_WINDOW = N * 150 s)")
    ap.add_argument("--leading-window", type=float, default=None,
                    help="Override LEADING_WINDOW in seconds")
    ap.add_argument("--smooth-window",  type=float, default=5 * 60,
                    help="Smooth window in seconds (default 300)")
    ap.add_argument("--lag-base", type=str, default=None,
                    help="Base directory for this lag level (auto-derived from --lag)")
    ap.add_argument("--questionnaire-dir", type=str, default=None)
    ap.add_argument("--metadata-file",     type=str, default=None)
    args = ap.parse_args()

    n   = args.lag
    lw  = args.leading_window if args.leading_window is not None else n * (args.smooth_window / 2)

    if args.lag_base is None:
        if n < 0:
            lag_name = f"lag_n{abs(n)}"
        elif n > 0:
            lag_name = f"lag_p{n}"
        else:
            lag_name = "lag_0"
        args.lag_base = os.path.join("processed_data", lag_name)

    run_merge(
        lag_base=args.lag_base,
        n=n,
        leading_window=lw,
        questionnaire_dir=args.questionnaire_dir,
        metadata_file=args.metadata_file,
    )
