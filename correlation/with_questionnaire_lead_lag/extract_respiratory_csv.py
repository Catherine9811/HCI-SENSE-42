"""
Lead-lag respiratory extractor.  See extract_behavioural_csv.py for
the window convention.
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_respiratory.rip import Resp
from data_definition import psydat_files
from data_parser import DataParser

from correlation.with_questionnaire.extract_respiratory_csv import (
    RespiratoryExhalationDurationExtractor,
    RespiratoryInhalationDurationExtractor,
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
        output_dir = os.path.join("processed_data", _lag_dir_name(n), "respiratory")

    half = smooth_window / 2.0
    outcome_definition = SleepinessAnswerExtractor()
    extractors = [
        RespiratoryInhalationDurationExtractor(),
        RespiratoryExhalationDurationExtractor(),
    ]
    processed: dict = {}

    for psydat_file in tqdm(psydat_files, desc=f"respiratory lw={leading_window:+.0f}s"):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        outcome_times, outcome_values = outcome_definition.process(parser)

        resp_path = os.path.join("..", "..", "data", "Respiration", f"P{participant_id:03}.wav")
        if not os.path.exists(resp_path):
            continue
        resp = Resp.from_wav(resp_path)
        resp.remove_baseline(method="savgol")
        resp.find_cycles(include_holds=True)

        for extractor in extractors:
            pred_times, pred_values = extractor.process(resp)
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
