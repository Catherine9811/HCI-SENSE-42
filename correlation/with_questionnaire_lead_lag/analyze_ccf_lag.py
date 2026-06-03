"""
Cross-lag CCF: alertness vs per-modality composites across lag levels.

Alertness is a composite questionnaire score computed as:

    alertness = effort + mental_demand + temporal_demand + frustration
                - performance + sleepiness

All six components are drawn from the questionnaire pivot columns that are
already present in the merged multimodal CSVs produced by run_all_lags.py.

For each lag step N (LEADING_WINDOW = N × 150 s, N = −6 … +6) this script:
  1. Loads the merged multimodal CSV for that lag from processed_data/lag_*/
  2. Derives the alertness score from the questionnaire columns.
  3. For each feature group (EEG, ECG, Resp, Behavioural, HCI, Proxy) builds a
     within-participant z-scored composite (mean of available group features).
  4. Computes the per-participant Pearson r between alertness and the composite.
  5. Aggregates across participants (mean ± 2000-resample bootstrap 95 % CI).

The result is a cross-lag correlation plot where the x-axis is the temporal
offset (LEADING_WINDOW in seconds) and the y-axis is the mean correlation.
A positive offset means the predictor window lies *after* the questionnaire;
a negative offset means the predictor window lies *before*.

  N = −1  →  offset = −150 s  →  window (−300, 0]   ← matches original extract
  N =  0  →  offset =    0 s  →  window (−150, +150] centred on questionnaire
  N = +1  →  offset = +150 s  →  window (0, +300]    purely after questionnaire

Outputs (written to <data_dir>/):
  ccf_lag_results.csv          columns: group, lag_n, leading_window_s,
                                         mean_r, ci_low, ci_high, n_participants
  ccf_lag_comparison.png       all-group overlay, offset in seconds on x-axis
  ccf_lag_grid.png             3×2 per-group panels

Usage:
  python analyze_ccf_lag.py
  python analyze_ccf_lag.py --data-dir /path/to/processed_data
  python analyze_ccf_lag.py --lags -3 -2 -1 0 1 2 3
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ═══════════════════════════════════════════════════════════════════════════════
# shared_config content (inlined — no external import needed)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS: Dict[str, List[str]] = {
    "physiological_eeg": [
        "alpha",
        # "beta",
        # "theta",
        # "delta",
        # "gamma",
    ],
    "physiological_ecg": [
        "cardiac_rr_interval_mean",
        "cardiac_rr_interval_var",
    ],
    "physiological_resp": [
        "respiratory_inhalation_duration_mean",
        "respiratory_inhalation_duration_var",
        "respiratory_exhalation_duration_mean",
        "respiratory_exhalation_duration_var",
    ],
    "behavioural": [
        # "head_pose_variation_mean",
        # "head_pose_movement_mean",
        "head_pitch_variation_mean",
        # "head_roll_variation_mean",
        # "head_yaw_variation_mean",
        "blink_times_mean",
        # "look_down_times_mean",
    ],
    "interaction_hci": [
        # "mouse_double_click_duration_mean",
        # "mouse_double_click_duration_var",
        # "mouse_double_click_movement_mean",
        # "mouse_double_click_movement_var",
        "mouse_double_click_distance_mean",
        # "mouse_double_click_distance_var",
        # "mouse_taskbar_navigation_efficiency_mean",
        # "mouse_taskbar_navigation_efficiency_var",
        # "mouse_toolbar_navigation_efficiency_mean",
        # "mouse_toolbar_navigation_efficiency_var",
        # "mouse_selection_coverage_mean",
        # "mouse_selection_coverage_var",
        # "mouse_folder_navigation_speed_mean",
        # "mouse_folder_navigation_speed_var",
        # "mouse_toolbar_navigation_speed_mean",
        # "mouse_toolbar_navigation_speed_var",
        # "mouse_confirm_dialog_duration_mean",
        # "mouse_confirm_dialog_duration_var",
        # "mouse_notification_duration_mean",
        # "mouse_notification_duration_var",
        # "mouse_open_folder_duration_mean",
        # "mouse_open_folder_duration_var",
        # "mouse_drag_folder_duration_mean",
        # "mouse_drag_folder_duration_var",
        # "mouse_close_window_duration_mean",
        # "mouse_close_window_duration_var",
        # "mouse_grouped_selection_duration_mean",
        # "mouse_grouped_selection_duration_var",
        # "mouse_open_notes_duration_mean",
        # "mouse_open_notes_duration_var",
        # "mouse_open_browser_duration_mean",
        # "mouse_open_browser_duration_var",
        # "mouse_open_file_manager_duration_mean",
        # "mouse_open_file_manager_duration_var",
        # "mouse_open_trash_bin_duration_mean",
        # "mouse_open_trash_bin_duration_var",
        # "mouse_open_folder_clicking_duration_mean",
        # "mouse_open_folder_clicking_duration_var",
        # "mouse_close_window_clicking_duration_mean",
        # "mouse_close_window_clicking_duration_var",
        # "mouse_close_window_unintended_clicks_mean",
        # "mouse_close_window_unintended_clicks_var",
        # "mouse_open_folder_unintended_clicks_mean",
        # "mouse_open_folder_unintended_clicks_var",
        # "mouse_close_to_toolbar_navigation_speed_mean",
        # "mouse_close_to_toolbar_navigation_speed_var",
        # "mouse_confirm_dialog_unintended_clicks_mean",
        # "mouse_confirm_dialog_unintended_clicks_var",
        # "mouse_open_notes_unintended_clicks_mean",
        # "mouse_open_notes_unintended_clicks_var",
        # "mouse_open_browser_unintended_clicks_mean",
        # "mouse_open_browser_unintended_clicks_var",
        # "mouse_open_file_manager_unintended_clicks_mean",
        # "mouse_open_file_manager_unintended_clicks_var",
        # "mouse_open_trash_bin_unintended_clicks_mean",
        # "mouse_open_trash_bin_unintended_clicks_var",
        # "mouse_open_notification_unintended_clicks_mean",
        # "mouse_open_notification_unintended_clicks_var",
        # "keyboard_shadow_typing_duration_mean",
        # "keyboard_shadow_typing_duration_var",
        # "keyboard_side_by_side_typing_duration_mean",
        # "keyboard_side_by_side_typing_duration_var",
        # "keyboard_shadow_typing_error_mean",
        # "keyboard_shadow_typing_error_var",
        # "keyboard_side_by_side_typing_error_mean",
        # "keyboard_side_by_side_typing_error_var",
        # "keyboard_typing_speed_mean",
        # "keyboard_typing_speed_var",
        # "keyboard_space_key_pressed_duration_mean",
        # "keyboard_space_key_pressed_duration_var",
        # "keyboard_space_key_typing_duration_mean",
        # "keyboard_space_key_typing_duration_var",
        "keyboard_pressed_duration_mean",
        # "keyboard_pressed_duration_var",
        # "keyboard_shadow_typing_efficiency_mean",
        # "keyboard_shadow_typing_efficiency_var",
        # "keyboard_side_by_side_typing_efficiency_mean",
        # "keyboard_side_by_side_typing_efficiency_var",
    ],
}

# Proxy: drag/drop error columns (subset of HCI used as behavioural ground-truth)
PROXY_ERROR_COLS: List[str] = [
    "mouse_drag_distance_mean",
    "mouse_drag_distance_var",
    "mouse_drop_distance_mean",
    "mouse_drop_distance_var",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis groups:  display name → feature column list
# ═══════════════════════════════════════════════════════════════════════════════

ANALYSIS_GROUPS: Dict[str, List[str]] = {
    "EEG":         list(FEATURE_GROUPS["physiological_eeg"]),
    "ECG":         list(FEATURE_GROUPS["physiological_ecg"]),
    "Resp":        list(FEATURE_GROUPS["physiological_resp"]),
    "Behavioural": list(FEATURE_GROUPS["behavioural"]),
    "HCI":         list(FEATURE_GROUPS["interaction_hci"]),
    "Proxy":       list(PROXY_ERROR_COLS),
}

# ── Visual style (mirrors analyze_multimodal_group_dynamics.py) ───────────────
GROUP_PALETTE: Dict[str, str] = {
    "EEG":         "#1f77b4",   # blue
    "ECG":         "#d62728",   # red
    "Resp":        "#9467bd",   # purple
    "Behavioural": "#2ca02c",   # green
    "HCI":         "#ff7f0e",   # orange
    "Proxy":       "#8c564b",   # brown
}
GROUP_MARKER: Dict[str, str] = {
    "EEG":         "o",
    "ECG":         "s",
    "Resp":        "^",
    "Behavioural": "v",
    "HCI":         "D",
    "Proxy":       "P",
}
GROUP_LINESTYLE: Dict[str, str] = {
    "EEG":         "-",
    "ECG":         "-",
    "Resp":        "-",
    "Behavioural": "-.",
    "HCI":         "--",
    "Proxy":       ":",
}

# ── Pipeline constants ────────────────────────────────────────────────────────
SMOOTH_WINDOW   = 5 * 60        # 300 s — window full-width used during extraction
LAG_STEP_S      = SMOOTH_WINDOW / 2  # 150 s per lag step
LAG_N_RANGE     = range(-6, 7)

# KEY is used only for finding the CSV files produced by run_all_lags.py
KEY             = "sleepiness"

# Alertness outcome: weighted sum of questionnaire columns
# alertness = effort + mental_demand + temporal_demand + frustration
#             − performance + sleepiness
ALERTNESS_COL        = "alertness"
ALERTNESS_COMPONENTS = {
    "effort":          +1.0,
    "mental_demand":   +1.0,
    "temporal_demand": +1.0,
    "frustration":     +1.0,
    "performance":     -1.0,
    "sleepiness":      +1.0,
}

PARTICIPANT_COL = "participant"
BOOTSTRAP_N     = 2000
RANDOM_STATE    = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Alertness score derivation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_alertness(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``alertness`` column derived from questionnaire component columns.

    alertness = effort + mental_demand + temporal_demand + frustration
                − performance + sleepiness

    Rows where any required component is NaN retain NaN in the alertness column.
    Components that are missing from the dataframe are skipped with a warning
    (so partial data still produces a result rather than all-NaN).
    """
    df = df.copy()
    available = {col: w for col, w in ALERTNESS_COMPONENTS.items() if col in df.columns}
    missing   = [col for col in ALERTNESS_COMPONENTS if col not in df.columns]

    if missing:
        print(f"  [alertness] Missing component column(s): {missing} — "
              "they will be excluded from the score.")

    if not available:
        raise ValueError(
            "No alertness component columns found in the dataframe. "
            "Ensure run_all_lags.py has been run with questionnaire extraction."
        )

    score = sum(df[col].astype(float) * w for col, w in available.items())
    df[ALERTNESS_COL] = score
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def lag_dir_name(n: int) -> str:
    if n < 0:
        return f"lag_n{abs(n)}"
    if n > 0:
        return f"lag_p{n}"
    return "lag_0"


def load_lag_dataframes(
    data_dir: str,
    lags: list[int],
    n_participants: int = 42,
) -> Dict[int, pd.DataFrame]:
    """Load the merged multimodal CSV for each requested lag step.

    First tries the per-lag file
    ``<data_dir>/lag_<N>/<n_participants>-sleepiness-multimodal.csv``.
    Falls back to the all-lags file if present.
    """
    fname = f"{n_participants}-{KEY}-multimodal.csv"
    all_lags_file = os.path.join(data_dir, f"{n_participants}-{KEY}-multimodal-all-lags.csv")

    # Try to use the all-lags file as a fast path
    all_lags_df: Optional[pd.DataFrame] = None
    if os.path.exists(all_lags_file):
        all_lags_df = pd.read_csv(all_lags_file)
        print(f"  Loaded all-lags file: {all_lags_file}  ({len(all_lags_df)} rows)")

    result: Dict[int, pd.DataFrame] = {}
    for n in lags:
        df: Optional[pd.DataFrame] = None

        if all_lags_df is not None and "lag_n" in all_lags_df.columns:
            sub = all_lags_df[all_lags_df["lag_n"] == n].copy()
            if len(sub) > 0:
                df = sub

        if df is None:
            per_lag_file = os.path.join(data_dir, lag_dir_name(n), fname)
            if os.path.exists(per_lag_file):
                df = pd.read_csv(per_lag_file)
                df["lag_n"] = n
                df["leading_window_s"] = n * LAG_STEP_S
            else:
                print(f"  [warn] Missing: {per_lag_file} — lag N={n:+d} skipped.")
                continue

        # Derive alertness score from questionnaire component columns
        df = compute_alertness(df)
        result[n] = df

    return result


def _zscore_within_participant(
    df: pd.DataFrame,
    feature_cols: List[str],
    pid_col: str = PARTICIPANT_COL,
) -> pd.DataFrame:
    """Return a copy of *df* with each *feature_col* z-scored per participant.
    Columns with zero variance within a participant are centred to zero."""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        for pid, idx in df.groupby(pid_col).groups.items():
            v = df.loc[idx, col].to_numpy(dtype=float)
            m = np.nanmean(v)
            s = np.nanstd(v)
            df.loc[idx, col] = (v - m) / s if s > 1e-9 else (v - m)
    return df


def build_composite(
    df: pd.DataFrame,
    feature_cols: List[str],
    pid_col: str = PARTICIPANT_COL,
) -> pd.Series:
    """Per-participant z-score each available feature, return their row-wise mean.

    Rows with no available feature become NaN.
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)

    zdf = _zscore_within_participant(df, available, pid_col)
    return zdf[available].mean(axis=1)


def _pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, returning NaN when fewer than 4 valid pairs or zero variance."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 4:
        return float("nan")
    a_v, b_v = a[mask], b[mask]
    if np.std(a_v) < 1e-9 or np.std(b_v) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a_v, b_v)[0, 1])


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = RANDOM_STATE,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean of *values* (NaN dropped)."""
    rng = np.random.RandomState(seed)
    v = values[~np.isnan(values)]
    if len(v) < 2:
        m = float(np.nanmean(v)) if len(v) else float("nan")
        return m, m
    boot = np.array([
        np.mean(v[rng.choice(len(v), len(v), replace=True)])
        for _ in range(n_boot)
    ])
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Core CCF computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lag_ccf(
    lag_dfs: Dict[int, pd.DataFrame],
    feature_cols: List[str],
    group_name: str,
    pid_col: str = PARTICIPANT_COL,
    outcome_col: str = ALERTNESS_COL,
) -> pd.DataFrame:
    """Compute mean Pearson r between *outcome_col* (alertness) and the group
    composite for every available lag level.

    Returns a DataFrame with columns:
        group, lag_n, leading_window_s, mean_r, ci_low, ci_high, n_participants
    """
    rows = []
    for n, df in sorted(lag_dfs.items()):
        lw = float(n * LAG_STEP_S)

        available = [c for c in feature_cols if c in df.columns]
        if not available:
            print(f"  [{group_name}] N={n:+d}: no features available — skipped.")
            continue

        composite = build_composite(df, available, pid_col)

        per_pid_r: List[float] = []
        for pid, g_idx in df.groupby(pid_col).groups.items():
            y  = df.loc[g_idx, outcome_col].to_numpy(dtype=float)
            x  = composite.loc[g_idx].to_numpy(dtype=float)
            r  = _pearsonr_safe(y, x)
            per_pid_r.append(r)

        r_arr = np.array(per_pid_r, dtype=float)
        valid  = r_arr[~np.isnan(r_arr)]
        mean_r = float(np.nanmean(r_arr))
        ci_lo, ci_hi = _bootstrap_ci(r_arr)

        rows.append({
            "group":            group_name,
            "lag_n":            int(n),
            "leading_window_s": lw,
            "mean_r":           mean_r,
            "ci_low":           ci_lo,
            "ci_high":          ci_hi,
            "n_participants":   int(len(valid)),
            "n_features_used":  int(len(available)),
        })
        print(f"  [{group_name:12s}] N={n:+d}  lw={lw:+5.0f}s  "
              f"r={mean_r:+.3f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
              f"n_pid={len(valid)}  feats={len(available)}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_group_ccf(
    ax: plt.Axes,
    ccf_df: pd.DataFrame,
    group: str,
    x_col: str = "leading_window_s",
) -> None:
    """Draw one group's CCF line + shaded CI onto *ax*."""
    sub = ccf_df[ccf_df["group"] == group].sort_values(x_col)
    if sub.empty:
        return
    x   = sub[x_col].to_numpy()
    r   = sub["mean_r"].to_numpy()
    lo  = sub["ci_low"].to_numpy()
    hi  = sub["ci_high"].to_numpy()
    c   = GROUP_PALETTE[group]

    ax.plot(x, r, linestyle=GROUP_LINESTYLE[group], marker=GROUP_MARKER[group],
            color=c, linewidth=1.9, markersize=6, label=group, zorder=3)
    ax.fill_between(x, lo, hi, color=c, alpha=0.15, zorder=2)

    # Mark lags where CI does not cross zero
    sig_mask = (lo > 0) | (hi < 0)
    if sig_mask.any():
        ax.scatter(x[sig_mask], r[sig_mask], s=140, facecolors="none",
                   edgecolors=c, linewidths=2.0, zorder=5)


def _decorate_ax(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    xticks: Optional[np.ndarray] = None,
    xticklabels: Optional[list] = None,
) -> None:
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
    ax.axvline(0.0, color="grey",  linestyle=":",  linewidth=0.9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title,   fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=8)


def plot_ccf_comparison(
    ccf_df: pd.DataFrame,
    out_path: str,
    x_col: str = "leading_window_s",
    x_label: str = "Predictor window offset  (seconds relative to questionnaire)",
) -> None:
    """Single-panel CCF comparison: all groups overlaid, x = time offset."""
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for group in ANALYSIS_GROUPS:
        _draw_group_ccf(ax, ccf_df, group, x_col=x_col)

    # Secondary x-axis ticks showing N values
    lags     = sorted(ccf_df["lag_n"].unique())
    offsets  = [n * LAG_STEP_S for n in lags]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(offsets)
    ax2.set_xticklabels([f"N={n:+d}" for n in lags], fontsize=7)
    ax2.set_xlabel("Lag step N  (LEADING_WINDOW = N × 150 s)", fontsize=8)

    _decorate_ax(
        ax,
        xlabel=x_label,
        ylabel="Mean Pearson r  (alertness vs composite)",
        title=(
            "Cross-lag CCF: alertness vs multimodal feature-group composites\n"
            "alertness = effort + mental_demand + temporal_demand + frustration − performance + sleepiness\n"
            "Shaded = 95 % bootstrap CI;  open marker = CI excludes zero"
        ),
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_ccf_grid(
    ccf_df: pd.DataFrame,
    out_path: str,
) -> None:
    """3×2 grid: one panel per modality group, both x-axis variants shown."""
    groups = list(ANALYSIS_GROUPS.keys())
    n_cols, n_rows = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.0, 7.5), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for i, group in enumerate(groups):
        ax = axes_flat[i]
        sub = ccf_df[ccf_df["group"] == group].sort_values("leading_window_s")
        if sub.empty:
            ax.set_visible(False)
            continue

        x  = sub["leading_window_s"].to_numpy()
        r  = sub["mean_r"].to_numpy()
        lo = sub["ci_low"].to_numpy()
        hi = sub["ci_high"].to_numpy()
        c  = GROUP_PALETTE[group]

        ax.plot(x, r, linestyle=GROUP_LINESTYLE[group], marker=GROUP_MARKER[group],
                color=c, linewidth=1.8, markersize=6)
        ax.fill_between(x, lo, hi, color=c, alpha=0.18)
        sig = (lo > 0) | (hi < 0)
        if sig.any():
            ax.scatter(x[sig], r[sig], s=120, facecolors="none", edgecolors=c,
                       linewidths=2.0, zorder=5)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(0, color="grey",  linestyle=":",  linewidth=0.8)
        ax.set_title(group, color=c, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Offset (s)", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("Mean Pearson r", fontsize=8)

    for j in range(len(groups), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Cross-lag CCF per modality group  (alertness vs composite)\n"
        "alertness = effort + mental_demand + temporal_demand + frustration − performance + sleepiness\n"
        "Shaded = 95 % bootstrap CI;  open marker = CI excludes zero",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(
    data_dir: str | None = None,
    lags: list[int] | None = None,
    n_participants: int = 42,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "processed_data"
        )
    if lags is None:
        lags = list(LAG_N_RANGE)

    print(f"\nData directory : {data_dir}")
    print(f"Lag steps (N)  : {lags}")
    print(f"SMOOTH_WINDOW  : {SMOOTH_WINDOW} s  |  LAG_STEP = {LAG_STEP_S:.0f} s")
    print(f"Groups         : {list(ANALYSIS_GROUPS.keys())}")
    print(f"Outcome        : {ALERTNESS_COL}  "
          f"= {' + '.join(f'({w:+.0f})×{c}' for c, w in ALERTNESS_COMPONENTS.items())}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n── Loading per-lag CSVs ─────────────────────────────────────────────")
    lag_dfs = load_lag_dataframes(data_dir, lags, n_participants=n_participants)
    if not lag_dfs:
        raise FileNotFoundError(
            f"No per-lag multimodal CSVs found in {data_dir}.\n"
            "Run run_all_lags.py first to generate the data."
        )
    print(f"  Loaded {len(lag_dfs)} lag level(s): "
          f"N in {sorted(lag_dfs.keys())}")

    # ── Compute CCF per group ─────────────────────────────────────────────────
    all_ccf: List[pd.DataFrame] = []
    for group_name, feature_cols in ANALYSIS_GROUPS.items():
        print(f"\n── Group: {group_name} "
              f"({len(feature_cols)} features defined) ─────────────────────────")
        ccf = compute_lag_ccf(lag_dfs, feature_cols, group_name)
        all_ccf.append(ccf)

    ccf_df = pd.concat(all_ccf, ignore_index=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out = os.path.join(data_dir, "ccf_lag_results.csv")
    ccf_df.to_csv(csv_out, index=False)
    print(f"\n── Saved results CSV → {csv_out}")
    print(ccf_df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating plots ──────────────────────────────────────────────────")
    plot_ccf_comparison(
        ccf_df,
        out_path=os.path.join(data_dir, "ccf_lag_comparison.png"),
        x_col="leading_window_s",
        x_label="Predictor window offset  (seconds relative to questionnaire)\n"
                "← predictor precedes questionnaire  |  predictor follows questionnaire →\n"
                f"Outcome: {ALERTNESS_COL} = "
                "effort + mental_demand + temporal_demand + frustration − performance + sleepiness",
    )
    plot_ccf_grid(
        ccf_df,
        out_path=os.path.join(data_dir, "ccf_lag_grid.png"),
    )

    return ccf_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the processed_data folder (default: ./processed_data)",
    )
    ap.add_argument(
        "--lags", type=int, nargs="+", default=None, metavar="N",
        help=f"Lag steps to include (default: {list(LAG_N_RANGE)})",
    )
    ap.add_argument(
        "--n-participants", type=int, default=42,
        help="Number of participants (used to find the right CSV filename)",
    )
    args = ap.parse_args()
    main(
        data_dir=args.data_dir,
        lags=args.lags,
        n_participants=args.n_participants,
    )
