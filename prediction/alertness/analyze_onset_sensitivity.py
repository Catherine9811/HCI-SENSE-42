from __future__ import annotations

"""
analyze_onset_sensitivity.py

Analysis 1 — Effect-size gradient across sleepiness severity.

Question
--------
Which modality first shows a detectable change as a participant transitions
from alert (Q1) to fatigued (Q4)?  The hypothesis is that EEG alpha requires
deep, established drowsiness before it responds reliably, while proxy errors
and physiological features detect the change earlier (at lower sleepiness).

Approach
--------
For each participant, raw sleepiness scores are ranked and divided into
within-participant quartiles:
  Q1 = most alert (bottom 25 % of their own sleepiness range)
  Q4 = most sleepy  (top    25 %)

Each feature is IQR-scaled within participant (matching the reference
prediction pipeline).  Cohen's d between Q1 (alert baseline) and each of
Q2, Q3, Q4 is computed per participant and then aggregated across
participants with 2000-resample bootstrap 95 % CI.

A positive d means the feature rises when sleepiness rises (expected for
alpha, proxy errors, respiratory duration).
A modality with large |d| already at Q2 detects mild onset.
A modality whose |d| only grows at Q4 needs established fatigue.

Active features
---------------
Exactly the uncommented features from the SHAP-selected FEATURE_GROUPS dict.
Two reference composites are also included for context:
  • Combined alertness (TLX + sleepiness) — expected large d at all Qs
  • TLX only — cognitive load without the sleepiness component

Outputs  →  prediction/alertness/processed_data/
  onset_sensitivity_d_results.csv           Cohen's d per feature/group/quartile
  onset_sensitivity_curves.png              Group composites overlaid (KEY figure)
  onset_sensitivity_heatmap.png             Feature × quartile d colour grid
  onset_sensitivity_group_breakdown.png     Per-group individual feature lines
  onset_sensitivity_summary_bar.png         First-to-medium-effect summary
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from prediction.alertness.predict_combined_alertness_lgbm_binary_participant_iqr_scaled import (
    GROUP_COL,
    SLEEPINESS_COL,
    TLX_COL,
    fit_participant_iqr_scaler,
    load_data,
    transform_with_participant_iqr_scaler,
)
from prediction.alertness.shared_config import DATA_PATH

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Active feature groups (uncommented SHAP-selected subset) ───────────────────
ACTIVE_FEATURE_GROUPS: Dict[str, List[str]] = {
    "EEG": [
        "alpha",
    ],
    "ECG": [
        "cardiac_rr_interval_mean",
        "cardiac_rr_interval_var",
    ],
    "Resp": [
        "respiratory_inhalation_duration_mean",
        "respiratory_inhalation_duration_var",
        "respiratory_exhalation_duration_mean",
        "respiratory_exhalation_duration_var",
    ],
    "Behavioural": [
        "head_pitch_variation_mean",
        # "head_yaw_variation_mean",
        "blink_times_mean",
    ],
    "HCI": [
        # "mouse_double_click_duration_var",
        "mouse_double_click_distance_mean",
        # "mouse_drag_folder_duration_mean",
        # "mouse_grouped_selection_duration_var",
        # "mouse_close_window_clicking_duration_mean",
        # "keyboard_space_key_pressed_duration_var",
        "keyboard_pressed_duration_mean",
    ],
    "Proxy": [
        "mouse_drag_distance_mean",
        "mouse_drag_distance_var",
        "mouse_drop_distance_mean",
        "mouse_drop_distance_var",
    ],
}

# Features whose natural direction is OPPOSITE to sleepiness (they DECREASE when
# the person becomes sleepier).  These correspond to the commented-out entries in
# ACTIVE_FEATURE_GROUPS above.  After IQR-scaling they are negated so that a
# positive d consistently means "moves in the sleepy direction."
INVERTED_FEATURE_GROUPS: Dict[str, List[str]] = {
    "Behavioural": [
        "head_yaw_variation_mean",
    ],
    "HCI": [
        # "mouse_drag_folder_duration_mean",
        # "mouse_grouped_selection_duration_var",
        # "mouse_close_window_clicking_duration_mean",
    ],
    "Proxy": [
        # "mouse_drop_distance_mean",
        # "mouse_drop_distance_var",
    ],
}
# Flat set for O(1) membership tests throughout the script
ALL_INVERTED: set = {
    f for cols in INVERTED_FEATURE_GROUPS.values() for f in cols
}

# Questionnaire modalities shown as individual sensor-equivalent lines
SLEEPINESS_KEY = "Sleepiness"       # raw KSS score
TLX_KEY        = "TLX"             # cognitive-load composite (no sleepiness)
# Combined reference ceiling (bin variable — expected large d by construction)
ALERTNESS_KEY  = "Alertness\n(TLX+sleep)"

# Ordered list used to drive plots: sensors first, then questionnaire, then ref
QUESTIONNAIRE_KEYS = [SLEEPINESS_KEY, TLX_KEY]

# Internal column names
QUARTILE_COL     = "__alertness_quartile__"   # now binned by combined alertness
ALERTNESS_BIN_COL = "__alertness_bin__"
COMPOSITE_SUFFIX  = "__composite__"

# ── Visual style ───────────────────────────────────────────────────────────────
GROUP_PALETTE: Dict[str, str] = {
    "EEG":          "#1f77b4",   # blue
    "ECG":          "#d62728",   # red
    "Resp":         "#9467bd",   # purple
    "Behavioural":  "#2ca02c",   # green
    "HCI":          "#ff7f0e",   # orange
    "Proxy":        "#8c564b",   # brown
    SLEEPINESS_KEY: "#e377c2",   # pink   ← questionnaire modality
    TLX_KEY:        "#17becf",   # cyan   ← questionnaire modality
    ALERTNESS_KEY:  "#bcbd22",   # olive  ← combined reference
}
GROUP_MARKER: Dict[str, str] = {
    "EEG": "o", "ECG": "s", "Resp": "^",
    "Behavioural": "v", "HCI": "D", "Proxy": "P",
    SLEEPINESS_KEY: "h", TLX_KEY: "H", ALERTNESS_KEY: "X",
}
GROUP_LS: Dict[str, str] = {
    "EEG": "-", "ECG": "-", "Resp": "-",
    "Behavioural": "-.", "HCI": "--", "Proxy": ":",
    SLEEPINESS_KEY: (0, (4, 1)),         # dash-dot-dot
    TLX_KEY:        (0, (3, 1, 1, 1)),   # dash-dot
    ALERTNESS_KEY:  (0, (1, 1)),         # dense dots (reference)
}

# Cohen's d thresholds (Cohen 1988)
D_SMALL  = 0.2
D_MEDIUM = 0.5
D_LARGE  = 0.8

N_QUARTILES  = 4
MIN_OBS_PID  = 8    # skip participant if fewer total valid rows
MIN_OBS_Q    = 2    # minimum per quartile to compute d
N_BOOT       = 2000
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def _iqr_scale(
    df: pd.DataFrame,
    cols: List[str],
    pid_col: str = GROUP_COL,
) -> pd.DataFrame:
    """IQR-scale *cols* within participant. Returns df copy."""
    available = [c for c in cols if c in df.columns]
    if not available:
        return df
    X      = df[available].copy()
    groups = df[pid_col]
    stats_ = fit_participant_iqr_scaler(X, groups, available)
    X_s    = transform_with_participant_iqr_scaler(X, groups, available, *stats_)
    out    = df.copy()
    out[available] = X_s[available]
    return out


def assign_quartiles(
    df: pd.DataFrame,
    bin_col: str      = SLEEPINESS_COL,
    pid_col: str      = GROUP_COL,
    n_quartiles: int  = N_QUARTILES,
    out_col: str      = QUARTILE_COL,
) -> pd.DataFrame:
    """Per-participant sleepiness quartiles (1 = alert, n_quartiles = sleepy)."""
    df  = df.copy()
    q   = np.full(len(df), np.nan)
    for pid, idx in df.groupby(pid_col).groups.items():
        idx  = np.asarray(list(idx))
        vals = df.loc[idx, bin_col].to_numpy(dtype=float)
        valid = ~np.isnan(vals)
        if valid.sum() < MIN_OBS_PID:
            continue
        edges  = np.quantile(vals[valid], np.linspace(0, 1, n_quartiles + 1))
        edges  = np.unique(edges)          # collapse ties
        if len(edges) < 2:
            continue
        labels = np.digitize(vals, bins=edges[1:-1]) + 1
        labels = np.clip(labels, 1, n_quartiles)
        q[idx] = labels
    df[out_col] = q
    return df


def build_composite(
    df: pd.DataFrame,
    feature_cols: List[str],
    composite_col: str,
) -> pd.DataFrame:
    """Row-wise mean of already-scaled columns → composite_col."""
    available = [c for c in feature_cols if c in df.columns]
    df = df.copy()
    df[composite_col] = df[available].mean(axis=1) if available else np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Core analysis: Cohen's d gradient
# ══════════════════════════════════════════════════════════════════════════════

def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g = bias-corrected Cohen's d.  Sign: positive when b > a.
    Returns NaN when either group has < MIN_OBS_Q observations or zero variance."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < MIN_OBS_Q or nb < MIN_OBS_Q:
        return float("nan")
    pooled_var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    if pooled_var < 1e-12:
        return float("nan")
    d = (np.mean(b) - np.mean(a)) / np.sqrt(pooled_var)
    # Hedges' correction for small samples
    correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return float(d * correction)


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int  = N_BOOT,
    seed: int    = RANDOM_STATE,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap percentile CI for the mean of *values* (NaN dropped)."""
    rng = np.random.RandomState(seed)
    v   = values[~np.isnan(values)]
    if v.size < 2:
        m = float(np.nanmean(v)) if v.size == 1 else float("nan")
        return m, m
    boot = np.array([
        np.mean(v[rng.choice(v.size, v.size, replace=True)])
        for _ in range(n_boot)
    ])
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def d_gradient_for_column(
    df: pd.DataFrame,
    feature_col: str,
    quartile_col: str    = QUARTILE_COL,
    pid_col: str         = GROUP_COL,
    n_quartiles: int     = N_QUARTILES,
) -> pd.DataFrame:
    """
    Per-participant Hedges' g(Q1, Qk) for k = 2 … n_quartiles.
    Returns DataFrame: quartile, mean_d, ci_low, ci_high, n_participants, median_d
    """
    per_pid: Dict[int, List[float]] = {q: [] for q in range(2, n_quartiles + 1)}

    for pid, idx in df.groupby(pid_col).groups.items():
        idx = np.asarray(list(idx))
        sub = df.loc[idx, [feature_col, quartile_col]].dropna()
        if len(sub) < MIN_OBS_PID:
            continue
        q1 = sub.loc[sub[quartile_col] == 1, feature_col].to_numpy(dtype=float)
        for q in range(2, n_quartiles + 1):
            qk = sub.loc[sub[quartile_col] == q, feature_col].to_numpy(dtype=float)
            per_pid[q].append(_hedges_g(q1, qk))

    rows = []
    for q in range(2, n_quartiles + 1):
        v  = np.array(per_pid[q], dtype=float)
        mn = float(np.nanmean(v))
        lo, hi = _bootstrap_ci(v)
        rows.append({
            "quartile":       q,
            "mean_d":         mn,
            "ci_low":         lo,
            "ci_high":        hi,
            "median_d":       float(np.nanmedian(v)),
            "n_participants": int(np.sum(~np.isnan(v))),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Run all gradients
# ══════════════════════════════════════════════════════════════════════════════

def run_all(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    1. Assign sleepiness quartiles on raw values.
    2. IQR-scale ALL sensor features (active + inverted) within participant.
    3. Negate inverted features so positive d = "moves in sleepy direction".
    4. Build group composites (active + inverted together).
    5. Compute d-gradient for every individual feature and every composite.
    6. Add reference composites (combined alertness, TLX-only).

    Returns (long-format results DataFrame, dict of per-group d-gradient DataFrames, df).
    """
    # Step 1 — compute combined alertness and use it as the binning variable
    df = df_raw.copy()
    df[ALERTNESS_BIN_COL] = df_raw[TLX_COL] + df_raw[SLEEPINESS_COL]
    df = assign_quartiles(df, bin_col=ALERTNESS_BIN_COL, out_col=QUARTILE_COL)

    # Step 2 — IQR-scale active AND inverted features together in one pass
    all_sensor_cols = sorted(
        {c for cols in ACTIVE_FEATURE_GROUPS.values()   for c in cols if c in df.columns} |
        {c for cols in INVERTED_FEATURE_GROUPS.values() for c in cols if c in df.columns}
    )
    df = _iqr_scale(df, all_sensor_cols)

    # Step 3 — negate inverted features after scaling
    negated: List[str] = []
    for feat in ALL_INVERTED:
        if feat in df.columns:
            df[feat] = -df[feat]
            negated.append(feat)
    if negated:
        print(f"  Negated {len(negated)} inverted feature(s): {negated}")

    # Step 4+5 — sensor group composites and per-feature gradients
    all_rows: List[dict] = []
    group_gradients: Dict[str, pd.DataFrame] = {}

    for group in ACTIVE_FEATURE_GROUPS:
        normal_feats   = [f for f in ACTIVE_FEATURE_GROUPS[group]            if f in df.columns]
        inverted_feats = [f for f in INVERTED_FEATURE_GROUPS.get(group, [])  if f in df.columns]
        all_feats      = normal_feats + inverted_feats
        if not all_feats:
            print(f"  [warn] {group}: no features found, skipping.")
            continue

        comp_col = f"{group}{COMPOSITE_SUFFIX}"
        df = build_composite(df, all_feats, comp_col)

        grad = d_gradient_for_column(df, comp_col, quartile_col=QUARTILE_COL)
        group_gradients[group] = grad
        for _, r in grad.iterrows():
            all_rows.append({"group": group, "feature": f"{group} composite",
                              "is_composite": True, "is_inverted": False, **r})

        for feat in normal_feats:
            grad_f = d_gradient_for_column(df, feat, quartile_col=QUARTILE_COL)
            for _, r in grad_f.iterrows():
                all_rows.append({"group": group, "feature": feat,
                                  "is_composite": False, "is_inverted": False, **r})

        for feat in inverted_feats:
            grad_f = d_gradient_for_column(df, feat, quartile_col=QUARTILE_COL)
            for _, r in grad_f.iterrows():
                all_rows.append({"group": group, "feature": feat,
                                  "is_composite": False, "is_inverted": True, **r})

    # Step 6 — questionnaire modalities (Sleepiness and TLX individually)
    # and combined alertness reference ceiling.
    # IQR-scale on a working copy so sensor columns are not disturbed.
    df_q = _iqr_scale(df, [SLEEPINESS_COL, TLX_COL, ALERTNESS_BIN_COL])

    for qname, qcol in [
        (SLEEPINESS_KEY, SLEEPINESS_COL),
        (TLX_KEY,        TLX_COL),
        (ALERTNESS_KEY,  ALERTNESS_BIN_COL),
    ]:
        if qcol not in df_q.columns:
            continue
        grad = d_gradient_for_column(df_q, qcol, quartile_col=QUARTILE_COL)
        group_gradients[qname] = grad
        for _, r in grad.iterrows():
            all_rows.append({"group": qname, "feature": qname,
                              "is_composite": True, "is_inverted": False, **r})

    return pd.DataFrame(all_rows), group_gradients, df


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _draw_threshold_lines(ax: plt.Axes, max_x: float = 4) -> None:
    for d, label, alpha in [
        (D_SMALL,  "small",  0.35),
        (D_MEDIUM, "medium", 0.50),
        (D_LARGE,  "large",  0.65),
    ]:
        ax.axhline(d,  color="grey", ls=":", lw=0.8, alpha=alpha)
        ax.axhline(-d, color="grey", ls=":", lw=0.8, alpha=alpha)
        ax.text(max_x + 0.05, d,  f"d={d}", va="center", fontsize=6.5,
                color="grey", alpha=alpha)


def _draw_group_curve(
    ax: plt.Axes,
    grad: pd.DataFrame,
    group: str,
    x_base: float = 1.0,    # Q1 anchor
) -> None:
    """Draw d-gradient line + CI band from Q1 (d=0 by definition) to Q4."""
    x   = [x_base] + grad["quartile"].tolist()       # Q1=1, Q2=2, Q3=3, Q4=4
    mn  = [0.0]    + grad["mean_d"].tolist()
    lo  = [0.0]    + grad["ci_low"].tolist()
    hi  = [0.0]    + grad["ci_high"].tolist()
    x, mn, lo, hi = np.array(x), np.array(mn), np.array(lo), np.array(hi)

    c  = GROUP_PALETTE.get(group, "black")
    m  = GROUP_MARKER.get(group, "o")
    ls = GROUP_LS.get(group, "-")

    ax.plot(x, mn, color=c, lw=2.0, ls=ls, marker=m, markersize=6,
            label=group, zorder=4)
    ax.fill_between(x, lo, hi, color=c, alpha=0.12, zorder=2)

    # Mark Q2 onset significance: open circle when CI excludes zero at Q2
    for xi, lo_i, hi_i, mn_i in zip(x[1:], lo[1:], hi[1:], mn[1:]):
        if lo_i > 0 or hi_i < 0:
            ax.scatter([xi], [mn_i], s=90, facecolors="none",
                       edgecolors=c, lw=1.8, zorder=5)


# ── Figure 1: Sensitivity curves (composite per group, all overlaid) ──────────

def plot_sensitivity_curves(
    group_gradients: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    sensor_groups = [g for g in ACTIVE_FEATURE_GROUPS if g in group_gradients]
    quest_groups  = [g for g in QUESTIONNAIRE_KEYS   if g in group_gradients]
    ref_groups    = [g for g in [ALERTNESS_KEY]       if g in group_gradients]

    for group in sensor_groups + quest_groups + ref_groups:
        _draw_group_curve(ax, group_gradients[group], group)

    _draw_threshold_lines(ax, max_x=4)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(1, color="lightgrey", lw=0.6)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(
        ["P0–25\n(alert\nbaseline)", "P25–50\n(mild)", "P50–75\n(moderate)", "P75–100\n(high\nalerting)"],
        fontsize=9,
    )
    ax.set_xlabel("Within-participant combined alertness percentile  (TLX + sleepiness)", fontsize=10)
    ax.set_ylabel("Hedges' g  (vs P0–25 alert baseline)", fontsize=10)
    ax.set_title(
        "Effect-size gradient: how much does each modality change\n"
        "as combined alertness increases from P0–25 (alert) to P75–100 (fatigued)?\n"
        "Shaded = 95 % bootstrap CI;  open marker = CI excludes zero",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(0.7, 4.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Per-group individual feature breakdown (2×3 panel grid) ─────────

def plot_group_breakdown(
    df: pd.DataFrame,
    results: pd.DataFrame,
    out_path: Path,
) -> None:
    # 6 sensor groups + 2 questionnaire + 1 reference = up to 9 panels; use 3×3
    groups = list(ACTIVE_FEATURE_GROUPS.keys()) + QUESTIONNAIRE_KEYS + [ALERTNESS_KEY]
    ncols, nrows = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 11),
                              sharey=False, sharex=True)
    axes_flat = axes.flatten()

    for i, group in enumerate(groups):
        ax    = axes_flat[i]
        color = GROUP_PALETTE.get(group, "black")
        normal_feats   = [f for f in ACTIVE_FEATURE_GROUPS.get(group, [])    if f in df.columns]
        inverted_feats = [f for f in INVERTED_FEATURE_GROUPS.get(group, [])  if f in df.columns]
        all_feats      = normal_feats + inverted_feats

        # individual feature lines — normal direction (dashed, medium alpha)
        for feat in normal_feats:
            sub = results[(results["group"] == group) & (results["feature"] == feat)
                          & ~results["is_inverted"]]
            if sub.empty:
                continue
            x  = [1] + sub["quartile"].tolist()
            mn = [0] + sub["mean_d"].tolist()
            ax.plot(x, mn, color=color, lw=1.0, alpha=0.50, ls="--")

        # individual feature lines — inverted direction (dotted, lighter)
        for feat in inverted_feats:
            sub = results[(results["group"] == group) & (results["feature"] == feat)
                          & results["is_inverted"]]
            if sub.empty:
                continue
            x  = [1] + sub["quartile"].tolist()
            mn = [0] + sub["mean_d"].tolist()
            ax.plot(x, mn, color=color, lw=1.0, alpha=0.40, ls=":")

        # composite line (thick, includes both normal + inverted)
        sub_comp = results[
            (results["group"] == group) & results["is_composite"]
        ]
        if not sub_comp.empty:
            x  = [1] + sub_comp["quartile"].tolist()
            mn = [0] + sub_comp["mean_d"].tolist()
            lo = [0] + sub_comp["ci_low"].tolist()
            hi = [0] + sub_comp["ci_high"].tolist()
            ax.plot(x, mn, color=color, lw=2.5, ls="-", label="composite", zorder=4)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, zorder=2)

        for d_th in [D_SMALL, D_MEDIUM]:
            ax.axhline( d_th, color="grey", ls=":", lw=0.7, alpha=0.5)
            ax.axhline(-d_th, color="grey", ls=":", lw=0.7, alpha=0.5)
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_title(group, fontsize=10, color=color, fontweight="bold")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=8)
        ax.set_ylabel("Hedges' g", fontsize=8)
        ax.grid(alpha=0.2)

        # Build legend handles
        handles = []
        for f in normal_feats:
            handles.append(plt.Line2D(
                [0], [0], color=color, lw=1, ls="--", alpha=0.6,
                label=f.replace("_", " "),
            ))
        for f in inverted_feats:
            handles.append(plt.Line2D(
                [0], [0], color=color, lw=1, ls=":", alpha=0.5,
                label=f"(−) {f.replace('_', ' ')}",
            ))
        handles.append(plt.Line2D(
            [0], [0], color=color, lw=2.5,
            label=f"composite ({len(all_feats)} feats)",
        ))
        ax.legend(handles=handles, fontsize=6.0, loc="best", framealpha=0.7)

    for j in range(len(groups), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Per-group individual feature sensitivity curves\n"
        "Thin dashed = individual features;  thick solid = group composite",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 3: Feature × quartile heatmap ──────────────────────────────────────

def plot_heatmap(
    results: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    One row per active feature (composites excluded), grouped by modality.
    Columns: Q2, Q3, Q4.  Cell colour = Hedges' g.
    """
    # Build ordered row list: composites first, then features (normal then inverted)
    order_rows: List[Tuple[str, str, bool, bool]] = []   # (group, feat, is_comp, is_inv)
    for group in list(ACTIVE_FEATURE_GROUPS.keys()) + QUESTIONNAIRE_KEYS + [ALERTNESS_KEY]:
        comp_rows = results[(results["group"] == group) & results["is_composite"]]
        if not comp_rows.empty:
            order_rows.append((group, comp_rows.iloc[0]["feature"], True, False))
        # Normal features first, then inverted
        for is_inv in [False, True]:
            feat_rows = results[
                (results["group"] == group) &
                ~results["is_composite"] &
                (results["is_inverted"] == is_inv)
            ]
            for feat in feat_rows["feature"].unique():
                order_rows.append((group, feat, False, is_inv))

    if not order_rows:
        return

    n_rows = len(order_rows)
    mat = np.full((n_rows, 3), np.nan)
    for i, (group, feat, is_comp, is_inv) in enumerate(order_rows):
        sub = results[
            (results["group"] == group) & (results["feature"] == feat)
        ].sort_values("quartile")
        for j, q in enumerate([2, 3, 4]):
            row = sub[sub["quartile"] == q]
            if not row.empty:
                mat[i, j] = row.iloc[0]["mean_d"]

    row_labels = []
    row_colors = []
    row_italics = []
    for (group, feat, is_comp, is_inv) in order_rows:
        if is_comp:
            label = (f"► {group} composite" if group in ACTIVE_FEATURE_GROUPS
                     else f"► {feat}")
        elif is_inv:
            label = f"(−) {feat}"   # prefix marks inverted direction
        else:
            label = feat
        row_labels.append(label.replace("_", " "))
        row_colors.append(GROUP_PALETTE.get(group, "black"))
        row_italics.append(is_inv)

    vmax = max(0.8, float(np.nanmax(np.abs(mat))))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(6, max(5, n_rows * 0.34)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Q2 (mild)", "Q3 (mod.)", "Q4 (high)"], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5)

    # colour y-tick labels; bold = composite, italic = inverted
    for tick, color, is_inv in zip(ax.get_yticklabels(), row_colors, row_italics):
        tick.set_color(color)
        text = tick.get_text()
        if "►" in text:
            tick.set_fontweight("bold")
        if is_inv:
            tick.set_style("italic")

    # annotate cells
    for i in range(n_rows):
        for j in range(3):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > 0.45 * vmax else "black")

    # group separator lines
    prev_group = None
    for i, (group, _, _, _) in enumerate(order_rows):
        if prev_group is not None and group != prev_group:
            ax.axhline(i - 0.5, color="white", lw=1.5)
        prev_group = group

    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Hedges' g  (vs Q1 baseline)", fontsize=8)
    for d in [D_SMALL, D_MEDIUM, D_LARGE]:
        cb.ax.axhline(d / vmax * 0.5 + 0.5, color="grey", lw=0.8, ls=":")
        cb.ax.axhline(-d / vmax * 0.5 + 0.5, color="grey", lw=0.8, ls=":")

    ax.set_title(
        "Effect-size heatmap: Hedges' g per feature and sleepiness quartile\n"
        "Red = rises with sleepiness  |  Blue = falls  |  (−) italic = sign inverted",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 4: Summary bar — first quartile at which |d| ≥ 0.5 ─────────────────

def plot_summary_bar(
    group_gradients: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """
    For each group composite, find the first quartile at which mean |d| ≥ 0.5
    (medium effect).  If it never reaches 0.5, record 'never' (shown as 5).
    Bar height = Q2 Hedges' g (the earliest onset indicator).
    """
    records = []
    for group, grad in group_gradients.items():
        q2_row = grad[grad["quartile"] == 2]
        q2_d   = float(q2_row["mean_d"].iloc[0]) if not q2_row.empty else 0.0
        q2_lo  = float(q2_row["ci_low"].iloc[0])  if not q2_row.empty else 0.0
        q2_hi  = float(q2_row["ci_high"].iloc[0]) if not q2_row.empty else 0.0

        # first quartile with |mean_d| >= medium
        first_q = next(
            (int(row["quartile"]) for _, row in grad.iterrows()
             if abs(row["mean_d"]) >= D_MEDIUM),
            None,
        )
        records.append({
            "group": group, "q2_d": q2_d, "q2_lo": q2_lo, "q2_hi": q2_hi,
            "first_medium_q": first_q,
        })

    rec = pd.DataFrame(records).sort_values("q2_d", ascending=False)

    fig, (ax_bar, ax_onset) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Q2 effect size bar chart
    colors = [GROUP_PALETTE.get(g, "black") for g in rec["group"]]
    x      = np.arange(len(rec))
    bars   = ax_bar.bar(x, rec["q2_d"], color=colors, alpha=0.85, width=0.55)
    ax_bar.errorbar(
        x, rec["q2_d"],
        yerr=[rec["q2_d"] - rec["q2_lo"], rec["q2_hi"] - rec["q2_d"]],
        fmt="none", ecolor="black", capsize=4, lw=1.2,
    )
    for bar, val in zip(bars, rec["q2_d"]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.01 if val >= 0 else val - 0.04,
                    f"{val:+.3f}", ha="center", va="bottom", fontsize=8)

    for d, label in [(D_SMALL, "small"), (D_MEDIUM, "medium"), (D_LARGE, "large")]:
        ax_bar.axhline(d,  color="grey", ls=":", lw=0.9, label=f"d={d} ({label})")
        ax_bar.axhline(-d, color="grey", ls=":", lw=0.9)
    ax_bar.axhline(0, color="black", lw=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(rec["group"], fontsize=9, rotation=20, ha="right")
    ax_bar.set_ylabel("Hedges' g at Q2 (mild sleepiness onset)", fontsize=9)
    ax_bar.set_title(
        "Onset sensitivity: effect size at Q2\n"
        "(how much does the modality change at mild sleepiness?)",
        fontsize=9,
    )
    ax_bar.legend(fontsize=7.5)
    ax_bar.grid(axis="y", alpha=0.3)

    # Right: first-quartile-to-medium-effect scatter / dot plot
    def _q_label(r) -> str:
        if r is None or (isinstance(r, float) and np.isnan(r)):
            return "Never"
        return {2: "Q2\n(mild)", 3: "Q3\n(mod.)", 4: "Q4\n(high)"}.get(int(r), "?")

    onset_labels = [_q_label(r) for r in rec["first_medium_q"]]
    jitter = np.random.RandomState(RANDOM_STATE).uniform(-0.12, 0.12, len(rec))
    y_pos  = [
        (4 - (int(r) if (r is not None and not (isinstance(r, float) and np.isnan(r))) else 4.8)) + j
        for r, j in zip(rec["first_medium_q"], jitter)
    ]

    for xi, (_, row) in enumerate(rec.iterrows()):
        grp = row["group"]
        yp  = y_pos[xi]
        ax_onset.scatter([yp], [xi], s=90,
                         color=GROUP_PALETTE.get(grp, "black"), zorder=4)
        ax_onset.text(yp + 0.05, xi, grp, va="center", fontsize=8)

    ax_onset.axvline(3.0, color="grey", ls=":", lw=0.8, label="Q2 threshold")
    ax_onset.axvline(2.0, color="grey", ls=":",  lw=0.8)
    ax_onset.axvline(1.0, color="grey", ls=":", lw=0.8)
    ax_onset.set_xticks([1, 2, 3])
    ax_onset.set_xticklabels(["Q4", "Q3", "Q2\n(onset)"], fontsize=9)
    ax_onset.set_xlabel("First quartile where |d| ≥ 0.5", fontsize=9)
    ax_onset.set_yticks([])
    ax_onset.set_title(
        "When does each modality first reach\nmedium effect size?",
        fontsize=9,
    )
    ax_onset.grid(axis="x", alpha=0.2)
    ax_onset.invert_xaxis()   # Q2 on right → "earliest" is rightmost

    fig.suptitle(
        "Onset sensitivity summary: which modality detects mild sleepiness earliest?",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 68)
    print("Analysis 1 — Onset sensitivity: effect-size gradient across")
    print("combined alertness percentiles (P0–25 alert → P75–100 fatigued)")
    print("=" * 68)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading data …")
    df_raw = load_data(DATA_PATH)
    alertness_bin = df_raw[TLX_COL] + df_raw[SLEEPINESS_COL]
    print(f"    {len(df_raw)} rows, {df_raw[GROUP_COL].nunique()} participants")
    print(f"    Combined alertness range: {alertness_bin.min():.1f} – {alertness_bin.max():.1f}")
    print(f"    Sleepiness range: {df_raw[SLEEPINESS_COL].min():.0f} – "
          f"{df_raw[SLEEPINESS_COL].max():.0f}")
    tmp = df_raw.copy()
    tmp[ALERTNESS_BIN_COL] = alertness_bin
    q_sizes = (
        assign_quartiles(tmp, bin_col=ALERTNESS_BIN_COL, out_col=QUARTILE_COL)
        .groupby(QUARTILE_COL).size()
    )
    print(f"    Alertness-percentile bin sizes:  "
          + "  ".join(f"P{int((q-1)*25)}–{int(q*25)}={n}" for q, n in q_sizes.items()))

    # ── Compute ───────────────────────────────────────────────────────────────
    print("\n[2] Computing Hedges' g gradient per feature and group composite …")
    results, group_gradients, df_scaled = run_all(df_raw)

    # Print summary table
    comp_results = results[results["is_composite"]].copy()
    all_groups = (list(ACTIVE_FEATURE_GROUPS.keys())
                  + QUESTIONNAIRE_KEYS + [ALERTNESS_KEY])
    print("\n    Group composite d-gradient  (mean Hedges' g ± 95 % CI):")
    print(f"    {'Group':26s}  {'P25–50':>16s}  {'P50–75':>16s}  {'P75–100':>16s}")
    for group in all_groups:
        sub = comp_results[comp_results["group"] == group].sort_values("quartile")
        if sub.empty:
            continue
        row_strs = [
            f"{r['mean_d']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
            for _, r in sub.iterrows()
        ]
        print(f"    {group:26s}  " + "  ".join(row_strs))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    print("\n[3] Saving results CSV …")
    csv_path = OUTPUT_DIR / "onset_sensitivity_d_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}  ({len(results)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[4] Generating plots …")
    plot_sensitivity_curves(
        group_gradients,
        out_path=OUTPUT_DIR / "onset_sensitivity_curves.png",
    )
    plot_group_breakdown(
        df_scaled, results,
        out_path=OUTPUT_DIR / "onset_sensitivity_group_breakdown.png",
    )
    plot_heatmap(
        results,
        out_path=OUTPUT_DIR / "onset_sensitivity_heatmap.png",
    )
    plot_summary_bar(
        group_gradients,
        out_path=OUTPUT_DIR / "onset_sensitivity_summary_bar.png",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
