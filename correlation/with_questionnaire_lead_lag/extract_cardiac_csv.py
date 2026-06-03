"""
Lead-lag cardiac (ECG) extractor.  See extract_behavioural_csv.py for
the window convention.
"""

import argparse
import os

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_definition import psydat_files
from data_parser import DataParser

from correlation.with_questionnaire.extract_cardiac_csv import (
    CardiacRRIntervalExtractor,
    filter_nan_indices,
)
from correlation.with_questionnaire_lead_lag.extract_behavioural_csv import (
    SleepinessAnswerExtractor,
)

# ── Window parameters ──────────────────────────────────────────────────────────
LEADING_WINDOW = 0
SMOOTH_WINDOW  = 5 * 60
# ──────────────────────────────────────────────────────────────────────────────


def run(
    leading_window: float = LEADING_WINDOW,
    smooth_window: float = SMOOTH_WINDOW,
    output_dir: str | None = None,
) -> None:
    if output_dir is None:
        from correlation.with_questionnaire_lead_lag.extract_behavioural_csv import _lag_dir_name
        n = round(leading_window / (smooth_window / 2))
        output_dir = os.path.join("processed_data", _lag_dir_name(n), "cardiac")

    half = smooth_window / 2.0
    outcome_definition = SleepinessAnswerExtractor()
    extractor = CardiacRRIntervalExtractor()
    processed: dict = {}

    for psydat_file in tqdm(psydat_files, desc=f"cardiac lw={leading_window:+.0f}s"):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        outcome_times, outcome_values = outcome_definition.process(parser)

        ecg_path = os.path.join("..", "..", "data", "ECG", f"P{participant_id:03}.fif")
        if not os.path.exists(ecg_path):
            continue
        ecg_file = mne.io.read_raw_fif(ecg_path, preload=True, verbose="WARNING")

        pred_times, pred_values = extractor.process(ecg_file)
        pred_times  = np.array(pred_times,  dtype=float)
        pred_values = np.array(pred_values, dtype=float)

        means, stds = [], []
        for ot in outcome_times:
            lo = ot + leading_window - half
            hi = ot + leading_window + half
            vals = pred_values[(pred_times > lo) & (pred_times <= hi)]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        processed.setdefault(f"{extractor.name}_mean", []).extend(means)
        processed.setdefault(f"{extractor.name}_var",  []).extend(stds)
        processed.setdefault("time",                   []).extend(outcome_times)
        processed.setdefault(outcome_definition.name,  []).extend(outcome_values)
        processed.setdefault("participant",             []).extend(
            [participant_id] * len(outcome_values))

    processed = filter_nan_indices(processed)
    df = pd.DataFrame(processed)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{len(psydat_files)}-{outcome_definition.name}.csv")
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--leading-window", type=float, default=None)
    ap.add_argument("--smooth-window",  type=float, default=SMOOTH_WINDOW)
    ap.add_argument("--output-dir",     type=str,   default=None)
    args = ap.parse_args()
    lw = args.leading_window if args.leading_window is not None else args.lag * (args.smooth_window / 2)
    run(leading_window=lw, smooth_window=args.smooth_window, output_dir=args.output_dir)
