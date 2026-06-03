"""
Run the full lead-lag extraction pipeline for lag steps N = -6 … +6.

Window convention
─────────────────
  SMOOTH_WINDOW  = 5 * 60   seconds   (300 s; full window width)
  LEADING_WINDOW = N * 150  seconds   (N * SMOOTH_WINDOW / 2)

  Association window for each outcome_time:
    (outcome_time + LEADING_WINDOW - SMOOTH_WINDOW/2,
     outcome_time + LEADING_WINDOW + SMOOTH_WINDOW/2]

  N = -1  →  window (-300, 0]  ← identical to the original with_questionnaire
  N =  0  →  window (-150, +150]  centred on outcome
  N = +1  →  window (0, +300]  immediately after outcome

Output layout
─────────────
  processed_data/
  ├── questionnaire/
  │   └── 42-questionnaires.csv          (extracted once, shared)
  ├── lag_n6/                            N = -6, lw = -900 s
  │   ├── behavioural/42-sleepiness.csv
  │   ├── cardiac/42-sleepiness.csv
  │   ├── event7to9/42-sleepiness.csv
  │   ├── respiratory/42-sleepiness.csv
  │   ├── webcam/42-sleepiness.csv
  │   └── 42-sleepiness-multimodal.csv
  ├── lag_n5/ … lag_0/ … lag_p6/        (same structure)
  └── 42-sleepiness-multimodal-all-lags.csv   (all lags concatenated)

Usage
─────
  python run_all_lags.py                      # N = -6 … +6
  python run_all_lags.py --lags -3 0 3        # specific lags only
  python run_all_lags.py --skip-questionnaire # if already extracted
  python run_all_lags.py --skip-extraction    # only re-merge
"""

import argparse
import os
import sys

import pandas as pd

# Ensure we can import relative to this script's directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Also ensure the project root is on the path
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Change CWD so that relative paths (../../data/…) inside each script work
os.chdir(_HERE)

# ── Parameters ─────────────────────────────────────────────────────────────────
LAG_N_RANGE   = range(-6, 7)   # N = -6, -5, …, 0, …, +5, +6
SMOOTH_WINDOW = 5 * 60         # 300 seconds
N_PARTICIPANTS = 42
KEY = "sleepiness"
# ──────────────────────────────────────────────────────────────────────────────


def lag_dir_name(n: int) -> str:
    """Return the sub-directory name for lag step N."""
    if n < 0:
        return f"lag_n{abs(n)}"
    if n > 0:
        return f"lag_p{n}"
    return "lag_0"


def leading_window_for(n: int) -> float:
    return n * (SMOOTH_WINDOW / 2)   # = N * 150 seconds


def run_all(
    lags: list[int] | None = None,
    skip_questionnaire: bool = False,
    skip_extraction: bool = False,
) -> None:
    if lags is None:
        lags = list(LAG_N_RANGE)

    # ── Step 1: Extract questionnaire responses (once, shared) ───────────────
    if not skip_questionnaire:
        print("\n" + "=" * 60)
        print("STEP 1 — Extracting questionnaire responses (shared)")
        print("=" * 60)
        from extract_questionnaire_csv import run as run_questionnaire
        run_questionnaire(
            output_dir=os.path.join("processed_data", "questionnaire"),
        )
    else:
        print("Skipping questionnaire extraction (--skip-questionnaire).")

    # ── Step 2: Per-lag extraction and merge ─────────────────────────────────
    from extract_behavioural_csv import run as run_behavioural
    from extract_cardiac_csv      import run as run_cardiac
    from extract_eeg_csv          import run as run_eeg
    from extract_respiratory_csv  import run as run_respiratory
    from extract_webcam_csv       import run as run_webcam
    from merge_multimodal_csv     import run_merge

    merged_paths: list[tuple[int, str]] = []

    for n in lags:
        lw   = leading_window_for(n)
        base = os.path.join("processed_data", lag_dir_name(n))

        print("\n" + "=" * 60)
        print(f"STEP 2  N = {n:+d}  |  LEADING_WINDOW = {lw:+.0f} s")
        print(f"  window = (outcome + {lw - SMOOTH_WINDOW/2:.0f} s, "
              f"outcome + {lw + SMOOTH_WINDOW/2:.0f} s]")
        print(f"  output base: {base}")
        print("=" * 60)

        if not skip_extraction:
            run_behavioural(
                leading_window=lw, smooth_window=SMOOTH_WINDOW,
                output_dir=os.path.join(base, "behavioural"),
            )
            run_cardiac(
                leading_window=lw, smooth_window=SMOOTH_WINDOW,
                output_dir=os.path.join(base, "cardiac"),
            )
            run_eeg(
                leading_window=lw, smooth_window=SMOOTH_WINDOW,
                output_dir=os.path.join(base, "event7to9"),
            )
            run_respiratory(
                leading_window=lw, smooth_window=SMOOTH_WINDOW,
                output_dir=os.path.join(base, "respiratory"),
            )
            run_webcam(
                leading_window=lw, smooth_window=SMOOTH_WINDOW,
                output_dir=os.path.join(base, "webcam"),
            )
        else:
            print("  Skipping extraction (--skip-extraction).")

        out = run_merge(
            lag_base=base,
            n=n,
            leading_window=lw,
            questionnaire_dir=os.path.join("processed_data", "questionnaire"),
        )
        if out:
            merged_paths.append((n, out))

    # ── Step 3: Concatenate all lags into one wide CSV ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Concatenating all lag levels")
    print("=" * 60)

    all_frames: list[pd.DataFrame] = []
    for n, path in merged_paths:
        if not os.path.exists(path):
            print(f"  Missing: {path} — skipping.")
            continue
        df = pd.read_csv(path)
        # Ensure lag columns are present (in case merge was run separately)
        df["lag_n"]           = n
        df["leading_window_s"] = leading_window_for(n)
        all_frames.append(df)
        print(f"  Loaded N={n:+d}: {len(df)} rows")

    if all_frames:
        all_df = pd.concat(all_frames, ignore_index=True)
        all_df.sort_values(by=["lag_n", "participant", "time"], inplace=True)
        all_out = os.path.join(
            "processed_data",
            f"{N_PARTICIPANTS}-{KEY}-multimodal-all-lags.csv",
        )
        all_df.to_csv(all_out, index=False)
        print(f"\n  All-lags CSV: {len(all_df)} rows × {len(all_df.columns)} cols")
        print(f"  Saved → {all_out}")
    else:
        print("  No per-lag CSVs found — all-lags merge skipped.")

    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--lags", type=int, nargs="+", default=None,
        metavar="N",
        help=f"Lag steps to process (default: {list(LAG_N_RANGE)})",
    )
    ap.add_argument(
        "--skip-questionnaire", action="store_true",
        help="Skip questionnaire extraction (use existing CSV)",
    )
    ap.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip per-modality extraction; only re-run merge and all-lags concat",
    )
    args = ap.parse_args()
    run_all(
        lags=args.lags,
        skip_questionnaire=args.skip_questionnaire,
        skip_extraction=args.skip_extraction,
    )
