"""
Lead-lag behavioural extractor.

Association window for each predictor relative to outcome_time:
    (outcome_time + LEADING_WINDOW - SMOOTH_WINDOW/2,
     outcome_time + LEADING_WINDOW + SMOOTH_WINDOW/2]

  LEADING_WINDOW = N * 150  (seconds; N is the lag step integer)
  SMOOTH_WINDOW  = 5 * 60   (seconds; window full-width)

  N = -1  →  window (-300, 0] relative to outcome  ← matches original
  N =  0  →  window (-150, +150] centred on outcome
  N = +1  →  window (0, +300] after outcome

Run standalone:
    python extract_behavioural_csv.py --lag -1
    python extract_behavioural_csv.py --leading-window -150
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_definition import psydat_files
from data_parser import DataParser

# Re-export all extractor classes from the canonical location so that
# other scripts in this folder can import them from here.
from correlation.with_questionnaire.extract_behavioural_csv import (  # noqa: F401
    AttentivenessAnswerExtractor,
    KeyboardPressedDurationExtractor,
    KeyboardShadowTypingDurationExtractor,
    KeyboardShadowTypingEfficiencyExtractor,
    KeyboardShadowTypingErrorExtractor,
    KeyboardSideBySideTypingDurationExtractor,
    KeyboardSideBySideTypingEfficiencyExtractor,
    KeyboardSideBySideTypingErrorExtractor,
    KeyboardSpaceKeyPressedDurationExtractor,
    KeyboardSpaceKeyTypingDurationExtractor,
    KeyboardTypingSpeedExtractor,
    MouseCloseToToolbarNavigationSpeedExtractor,
    MouseCloseWindowClickingDurationExtractor,
    MouseCloseWindowDurationExtractor,
    MouseCloseWindowUnintendedClicksExtractor,
    MouseConfirmDialogDurationExtractor,
    MouseConfirmDialogUnintendedClicksExtractor,
    MouseDoubleClickDistanceExtractor,
    MouseDoubleClickDurationExtractor,
    MouseDoubleClickMovementExtractor,
    MouseDragDistanceExtractor,
    MouseDragFolderDurationExtractor,
    MouseDropDistanceExtractor,
    MouseFolderNavigationSpeedExtractor,
    MouseGroupedSelectionDurationExtractor,
    MouseNotificationDurationExtractor,
    MouseOpenBrowserDurationExtractor,
    MouseOpenBrowserUnintendedClicksExtractor,
    MouseOpenFileManagerDurationExtractor,
    MouseOpenFileManagerUnintendedClicksExtractor,
    MouseOpenFolderClickingDurationExtractor,
    MouseOpenFolderDurationExtractor,
    MouseOpenFolderUnintendedClicksExtractor,
    MouseOpenNotesDurationExtractor,
    MouseOpenNotesUnintendedClicksExtractor,
    MouseOpenNotificationUnintendedClicksExtractor,
    MouseOpenTrashBinDurationExtractor,
    MouseOpenTrashBinUnintendedClicksExtractor,
    MouseSelectionCoverageExtractor,
    MouseTaskbarNavigationEfficiencyExtractor,
    MouseToolbarNavigationEfficiencyExtractor,
    MouseToolbarNavigationSpeedExtractor,
    PerformanceAnswerExtractor,
    SleepinessAnswerExtractor,
    TemporalDemandAnswerExtractor,
    filter_nan_indices,
    filter_time_series,
)

# ── Window parameters ──────────────────────────────────────────────────────────
LEADING_WINDOW = 0        # seconds; set to N * 150 for lead-lag step N
SMOOTH_WINDOW  = 5 * 60   # seconds; window = (center - SW/2, center + SW/2]
# ──────────────────────────────────────────────────────────────────────────────

_PREDICTOR_DEFINITIONS = [
    MouseDoubleClickDistanceExtractor(),
    MouseDoubleClickDurationExtractor(),
    MouseDoubleClickMovementExtractor(),
    MouseDragDistanceExtractor(),
    MouseDropDistanceExtractor(),
    MouseTaskbarNavigationEfficiencyExtractor(),
    MouseToolbarNavigationEfficiencyExtractor(),
    MouseSelectionCoverageExtractor(),
    MouseFolderNavigationSpeedExtractor(),
    MouseToolbarNavigationSpeedExtractor(),
    MouseConfirmDialogDurationExtractor(),
    MouseNotificationDurationExtractor(),
    MouseOpenFolderDurationExtractor(),
    MouseDragFolderDurationExtractor(),
    MouseCloseWindowDurationExtractor(),
    MouseGroupedSelectionDurationExtractor(),
    MouseOpenNotesDurationExtractor(),
    MouseOpenBrowserDurationExtractor(),
    MouseOpenFileManagerDurationExtractor(),
    MouseOpenTrashBinDurationExtractor(),
    KeyboardShadowTypingDurationExtractor(),
    KeyboardSideBySideTypingDurationExtractor(),
    KeyboardShadowTypingErrorExtractor(),
    KeyboardSideBySideTypingErrorExtractor(),
    KeyboardTypingSpeedExtractor(),
    KeyboardSpaceKeyPressedDurationExtractor(),
    KeyboardSpaceKeyTypingDurationExtractor(),
    KeyboardPressedDurationExtractor(),
    KeyboardShadowTypingEfficiencyExtractor(),
    KeyboardSideBySideTypingEfficiencyExtractor(),
    MouseOpenFolderClickingDurationExtractor(),
    MouseCloseWindowClickingDurationExtractor(),
    MouseCloseWindowUnintendedClicksExtractor(),
    MouseOpenFolderUnintendedClicksExtractor(),
    MouseCloseToToolbarNavigationSpeedExtractor(),
    MouseConfirmDialogUnintendedClicksExtractor(),
    MouseOpenNotesUnintendedClicksExtractor(),
    MouseOpenBrowserUnintendedClicksExtractor(),
    MouseOpenFileManagerUnintendedClicksExtractor(),
    MouseOpenTrashBinUnintendedClicksExtractor(),
    MouseOpenNotificationUnintendedClicksExtractor(),
]


def _lag_dir_name(n: int) -> str:
    if n < 0:
        return f"lag_n{abs(n)}"
    if n > 0:
        return f"lag_p{n}"
    return "lag_0"


def run(
    leading_window: float = LEADING_WINDOW,
    smooth_window: float = SMOOTH_WINDOW,
    output_dir: str | None = None,
) -> None:
    """Extract behavioural features with a lead-lag window and save to CSV."""
    if output_dir is None:
        n = round(leading_window / (smooth_window / 2))
        output_dir = os.path.join("processed_data", _lag_dir_name(n), "behavioural")

    half = smooth_window / 2.0
    outcome_definition = SleepinessAnswerExtractor()
    processed: dict = {}

    for psydat_file in tqdm(psydat_files, desc=f"behavioural lw={leading_window:+.0f}s"):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        outcome_times, outcome_values = outcome_definition.process(parser)
        outcome_times, outcome_values = filter_time_series(outcome_times, outcome_values)

        for extractor in _PREDICTOR_DEFINITIONS:
            pred_times, pred_values = extractor.process(parser)
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lag", type=int, default=0,
                    help="Lag step N; LEADING_WINDOW = N * SMOOTH_WINDOW/2 seconds")
    ap.add_argument("--leading-window", type=float, default=None,
                    help="Override: LEADING_WINDOW in seconds directly")
    ap.add_argument("--smooth-window",  type=float, default=SMOOTH_WINDOW,
                    help=f"Window full-width in seconds (default {SMOOTH_WINDOW})")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    lw = args.leading_window if args.leading_window is not None else args.lag * (args.smooth_window / 2)
    run(leading_window=lw, smooth_window=args.smooth_window, output_dir=args.output_dir)
