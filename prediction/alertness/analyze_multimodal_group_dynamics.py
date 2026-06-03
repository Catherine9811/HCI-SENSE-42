from __future__ import annotations

"""
Extend the subjective-proxy dynamics analysis to compare SIX feature-group
composites simultaneously against the combined alertness score (TLX + sleepiness):

  1. EEG            — electroencephalography band-power features
  2. ECG            — cardiac RR-interval features
  3. Resp           — respiratory duration features
  4. HCI            — mouse / keyboard interaction features
  5. Behavioural    — head-pose, blink, and gaze features
  6. HCI (Proxy)    — proxy error columns (drag/drop distance) as HCI ground-truth

The same four analyses are run per group (CCF, event study, distributed-lag,
Granger causality).  Results are overlaid on shared comparison figures so all
six modality types can be compared side-by-side.

Comparison figures:
  - multimodal_ccf_comparison.png
  - multimodal_event_proxy_to_alertness_comparison.png
  - multimodal_event_alertness_to_proxy_comparison.png
  - multimodal_dl_comparison.png
  - multimodal_granger_comparison.png
  - multimodal_summary_panel.png            (2×2 overview)

Per-group CSV outputs (prefix = multimodal_<group_slug>_):
  - ccf_per_pid.csv
  - ccf_aggregated.csv
  - event_composite_to_alertness_summary.csv
  - event_alertness_to_composite_summary.csv
  - dl_coefficients.csv
  - granger_per_pid.csv
  - granger_summary.csv
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prediction.alertness.analyze_subjective_proxy_dynamics import (
    ALERTNESS_SCORE_COL,
    BOOTSTRAP_N,
    MAX_LAG,
    N_PERMUTATIONS,
    ORDER_COL,
    RANDOM_STATE,
    attach_alertness_score,
    compute_ccf_per_pid,
    distributed_lag_regression,
    event_study,
    event_study_summary,
    granger_causality_per_pid,
    granger_summary,
    per_pid_zscored_series,
    permutation_test_ccf,
)
from prediction.alertness.predict_combined_alertness_lgbm_binary_participant_iqr_scaled import (
    GROUP_COL,
    load_data,
)
from prediction.alertness.shared_config import DATA_PATH, FEATURE_GROUPS, PROXY_ERROR_COLS


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "processed_data"

# Map display name → resolved list of feature column names.
# EEG/ECG/Resp/HCI/Behavioural draw from FEATURE_GROUPS; HCI (Proxy) uses
# PROXY_ERROR_COLS directly to serve as the HCI behavioural ground-truth.
ANALYSIS_GROUPS: Dict[str, List[str]] = {
    "EEG":         list(FEATURE_GROUPS["physiological_eeg"]),
    "ECG":         list(FEATURE_GROUPS["physiological_ecg"]),
    "Resp":        list(FEATURE_GROUPS["physiological_resp"]),
    "HCI":         list(FEATURE_GROUPS["interaction_hci"]),
    "Behavioural": list(FEATURE_GROUPS["behavioural"]),
    "HCI (Proxy)": list(PROXY_ERROR_COLS),
}

GROUP_PALETTE: Dict[str, str] = {
    "EEG":         "#1f77b4",  # blue
    "ECG":         "#d62728",  # red
    "Resp":        "#9467bd",  # purple
    "HCI":         "#ff7f0e",  # orange
    "Behavioural": "#2ca02c",  # green
    "HCI (Proxy)": "#8c564b",  # brown
}
GROUP_MARKER: Dict[str, str] = {
    "EEG":         "o",
    "ECG":         "s",
    "Resp":        "^",
    "HCI":         "D",
    "Behavioural": "v",
    "HCI (Proxy)": "P",
}
GROUP_LINESTYLE: Dict[str, str] = {
    "EEG":         "-",
    "ECG":         "-",
    "Resp":        "-",
    "HCI":         "--",
    "Behavioural": "-.",
    "HCI (Proxy)": ":",
}


# ─────────────────────────────────────────────────────────────────────────────
# Composite construction
# ─────────────────────────────────────────────────────────────────────────────

def _collect_group_features(group_keys: List[str]) -> List[str]:
    """Merge feature lists from multiple FEATURE_GROUPS entries, deduplicating."""
    seen: set = set()
    out: List[str] = []
    for k in group_keys:
        for f in FEATURE_GROUPS.get(k, []):
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def build_group_composite(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: List[str],
    composite_col: str,
) -> pd.DataFrame:
    """Per-participant z-score each available feature, then assign their mean
    to *composite_col*.  Rows where no feature is available become NaN."""
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        df = df.copy()
        df[composite_col] = np.nan
        return df

    df = df.copy()
    # z-score every feature within each participant
    for c in available:
        for pid, idx in df.groupby(group_col).groups.items():
            v = df.loc[idx, c].to_numpy(dtype=float)
            m, s = np.nanmean(v), np.nanstd(v) + 1e-9
            df.loc[idx, c] = (v - m) / s

    df[composite_col] = df[available].mean(axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_ccf_line(
    ax: plt.Axes,
    perm_df: pd.DataFrame,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    """Draw one CCF curve (mean ± shaded 95% CI) onto *ax*."""
    if perm_df.empty:
        return
    lags = perm_df["lag"].to_numpy()
    r = perm_df["mean_r"].to_numpy()
    lo = perm_df["ci_low"].to_numpy()
    hi = perm_df["ci_high"].to_numpy()

    ax.plot(lags, r, linestyle=linestyle, marker=marker, color=color,
            linewidth=1.8, markersize=6, label=label)
    ax.fill_between(lags, lo, hi, color=color, alpha=0.15)

    if "p_perm" in perm_df.columns:
        sig_mask = perm_df["p_perm"].to_numpy() < 0.05
        if sig_mask.any():
            ax.scatter(lags[sig_mask], r[sig_mask], s=130, facecolors="none",
                       edgecolors=color, linewidths=2.0, zorder=5)


def _draw_event_line(
    ax: plt.Axes,
    summary: pd.DataFrame,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    """Draw one event-study curve onto *ax*."""
    if summary.empty:
        return
    offsets = summary["offset"].to_numpy()
    means = summary["mean"].to_numpy()
    lo = summary["ci_low"].to_numpy()
    hi = summary["ci_high"].to_numpy()

    ax.plot(offsets, means, linestyle=linestyle, marker=marker, color=color,
            linewidth=1.8, markersize=6, label=label)
    ax.fill_between(offsets, lo, hi, color=color, alpha=0.15)

    if "p_perm" in summary.columns:
        sig_mask = summary["p_perm"].to_numpy() < 0.05
        if sig_mask.any():
            ax.scatter(offsets[sig_mask], means[sig_mask], s=130,
                       facecolors="none", edgecolors=color, linewidths=2.0,
                       zorder=5)


def _draw_dl_line(
    ax: plt.Axes,
    coefs: pd.DataFrame,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    """Draw one DL coefficient curve onto *ax*."""
    if coefs.empty:
        return
    lags = coefs["lag"].to_numpy()
    c = coefs["coef"].to_numpy()
    lo = coefs["ci_low"].to_numpy()
    hi = coefs["ci_high"].to_numpy()

    ax.plot(lags, c, linestyle=linestyle, marker=marker, color=color,
            linewidth=1.8, markersize=6, label=label)
    ax.fill_between(lags, lo, hi, color=color, alpha=0.15)

    if "p_bonferroni" in coefs.columns:
        sig_mask = coefs["p_bonferroni"].to_numpy() < 0.05
        if sig_mask.any():
            ax.scatter(lags[sig_mask], c[sig_mask], s=130,
                       facecolors="none", edgecolors=color, linewidths=2.0,
                       zorder=5)


def _draw_granger_line(
    ax: plt.Axes,
    summary: pd.DataFrame,
    direction: str,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    """Draw proportion-significant Granger curve for *direction* onto *ax*."""
    if summary.empty:
        return
    sub = summary[summary["direction"] == direction].copy()
    if sub.empty:
        return
    sub = sub.sort_values("lag")
    total = sub["n_total"].clip(lower=1)
    prop = sub["n_sig_005"] / total

    ax.plot(sub["lag"], prop, linestyle=linestyle, marker=marker,
            color=color, linewidth=1.8, markersize=6, label=label)


def _finalise_ax(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    hline: bool = True,
    vline: bool = True,
) -> None:
    if hline:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
    if vline:
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level comparison figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccf_comparison(
    results: Dict[str, Dict],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for group, res in results.items():
        _draw_ccf_line(
            ax, res["ccf_perm"], label=group,
            color=GROUP_PALETTE[group],
            marker=GROUP_MARKER[group],
            linestyle=GROUP_LINESTYLE[group],
        )
    _finalise_ax(
        ax,
        xlabel="Lag k  (k>0 → alertness leads group composite;  k<0 → composite leads alertness)",
        ylabel="Mean Pearson r across participants",
        title="Cross-correlation: alertness vs feature-group composites\n"
              "(shaded = 95% bootstrap CI; open marker = p_perm < 0.05)",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_event_comparison(
    results: Dict[str, Dict],
    direction: str,
    out_path: Path,
) -> None:
    """direction: 'composite_to_alertness' or 'alertness_to_composite'."""
    if direction == "composite_to_alertness":
        key = "event_comp_to_alert"
        anchor_label = "composite spike"
        target_ylabel = "Alertness z (mean ± 95% CI)"
        title = "Event study: alertness trajectory around composite spikes"
    else:
        key = "event_alert_to_comp"
        anchor_label = "alertness spike"
        target_ylabel = "Composite z (mean ± 95% CI)"
        title = "Event study: composite trajectory around alertness spikes"

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for group, res in results.items():
        _draw_event_line(
            ax, res[key], label=group,
            color=GROUP_PALETTE[group],
            marker=GROUP_MARKER[group],
            linestyle=GROUP_LINESTYLE[group],
        )
    _finalise_ax(
        ax,
        xlabel=f"Offset from {anchor_label} (trials; <0 = before, >0 = after)",
        ylabel=target_ylabel,
        title=title + "\n(shaded = 95% CI; open marker = p_perm < 0.05)",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_dl_comparison(
    results: Dict[str, Dict],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for group, res in results.items():
        _draw_dl_line(
            ax, res["dl"], label=group,
            color=GROUP_PALETTE[group],
            marker=GROUP_MARKER[group],
            linestyle=GROUP_LINESTYLE[group],
        )
    _finalise_ax(
        ax,
        xlabel="Lag k of alertness in regression (k<0 → alertness leads composite; "
               "k>0 → composite leads alertness)",
        ylabel="β_k (cluster-robust 95% CI)",
        title="Distributed-lag β: composite ~ alertness_{t+k}\n"
              "(shaded = 95% CI; open marker = Bonferroni p < 0.05)",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_granger_comparison(
    results: Dict[str, Dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    directions = ["alertness→composite", "composite→alertness"]
    titles = [
        "alertness → composite  (prop. participants p<0.05)",
        "composite → alertness  (prop. participants p<0.05)",
    ]
    for ax, direction, title in zip(axes, directions, titles):
        for group, res in results.items():
            _draw_granger_line(
                ax, res["granger_summary"], direction=direction, label=group,
                color=GROUP_PALETTE[group],
                marker=GROUP_MARKER[group],
                linestyle=GROUP_LINESTYLE[group],
            )
        ax.axhline(0.05, color="red", linestyle=":", linewidth=0.8,
                   label="p=0.05 reference")
        ax.set_xlabel("Lag (trials)", fontsize=9)
        ax.set_ylabel("Proportion sig. participants", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Granger causality: alertness ↔ feature-group composites",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_summary_panel(
    results: Dict[str, Dict],
    out_path: Path,
) -> None:
    """2×2 panel: CCF | event proxy→alert | event alert→proxy | DL."""
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))

    # [0,0] CCF
    ax = axes[0, 0]
    for group, res in results.items():
        _draw_ccf_line(ax, res["ccf_perm"], label=group,
                       color=GROUP_PALETTE[group],
                       marker=GROUP_MARKER[group],
                       linestyle=GROUP_LINESTYLE[group])
    _finalise_ax(ax,
                 xlabel="Lag k  (k>0: alertness leads)",
                 ylabel="Mean Pearson r",
                 title="A. Cross-correlation (CCF)")

    # [0,1] Event: composite → alertness
    ax = axes[0, 1]
    for group, res in results.items():
        _draw_event_line(ax, res["event_comp_to_alert"], label=group,
                         color=GROUP_PALETTE[group],
                         marker=GROUP_MARKER[group],
                         linestyle=GROUP_LINESTYLE[group])
    _finalise_ax(ax,
                 xlabel="Offset from composite spike (trials)",
                 ylabel="Alertness z",
                 title="B. Event study: alertness around composite spikes")

    # [1,0] Event: alertness → composite
    ax = axes[1, 0]
    for group, res in results.items():
        _draw_event_line(ax, res["event_alert_to_comp"], label=group,
                         color=GROUP_PALETTE[group],
                         marker=GROUP_MARKER[group],
                         linestyle=GROUP_LINESTYLE[group])
    _finalise_ax(ax,
                 xlabel="Offset from alertness spike (trials)",
                 ylabel="Composite z",
                 title="C. Event study: composite around alertness spikes")

    # [1,1] DL
    ax = axes[1, 1]
    for group, res in results.items():
        _draw_dl_line(ax, res["dl"], label=group,
                      color=GROUP_PALETTE[group],
                      marker=GROUP_MARKER[group],
                      linestyle=GROUP_LINESTYLE[group])
    _finalise_ax(ax,
                 xlabel="Lag k (k<0: alertness leads composite)",
                 ylabel="β_k",
                 title="D. Distributed-lag regression")

    fig.suptitle(
        "Multimodal temporal dynamics: alertness vs EEG / ECG / Resp / HCI / Behavioural / HCI (Proxy)\n"
        "(shaded = 95% CI; open marker = significant)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-group analysis runner
# ─────────────────────────────────────────────────────────────────────────────

def run_group_analysis(
    df: pd.DataFrame,
    group_name: str,
    composite_col: str,
    max_lag: int,
    n_perm: int,
) -> Dict:
    slug = group_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n{'─' * 60}")
    print(f"  Group: {group_name}  (composite col: {composite_col})")
    print(f"{'─' * 60}")

    alertness_series = per_pid_zscored_series(df, GROUP_COL, ORDER_COL, ALERTNESS_SCORE_COL)
    composite_series = per_pid_zscored_series(df, GROUP_COL, ORDER_COL, composite_col)

    n_valid = sum(1 for v in composite_series.values() if not np.all(np.isnan(v)))
    print(f"  Participants with non-NaN composite: {n_valid}/{len(composite_series)}")

    # A. CCF
    print(f"  [A] CCF + permutation test (N={n_perm})")
    ccf_pid = compute_ccf_per_pid(alertness_series, composite_series, max_lag)
    ccf_pid.to_csv(
        OUTPUT_DIR / f"multimodal_{slug}_ccf_per_pid.csv", index=False)
    perm_df = permutation_test_ccf(alertness_series, composite_series,
                                   max_lag, n_perm=n_perm)
    perm_df.to_csv(
        OUTPUT_DIR / f"multimodal_{slug}_ccf_aggregated.csv", index=False)

    # B. Event study (composite → alertness, alertness → composite)
    print("  [B] Event study")
    es_c2a = event_study(df, GROUP_COL, ORDER_COL,
                         composite_col, ALERTNESS_SCORE_COL, max_lag)
    s_c2a = event_study_summary(es_c2a)
    if not s_c2a.empty:
        s_c2a.to_csv(
            OUTPUT_DIR / f"multimodal_{slug}_event_composite_to_alertness_summary.csv",
            index=False)

    es_a2c = event_study(df, GROUP_COL, ORDER_COL,
                         ALERTNESS_SCORE_COL, composite_col, max_lag)
    s_a2c = event_study_summary(es_a2c)
    if not s_a2c.empty:
        s_a2c.to_csv(
            OUTPUT_DIR / f"multimodal_{slug}_event_alertness_to_composite_summary.csv",
            index=False)

    # C. Distributed-lag regression
    print("  [C] Distributed-lag regression")
    dl = distributed_lag_regression(df, GROUP_COL, ORDER_COL,
                                    ALERTNESS_SCORE_COL, composite_col, max_lag)
    if not dl.empty:
        dl.to_csv(
            OUTPUT_DIR / f"multimodal_{slug}_dl_coefficients.csv", index=False)

    # D. Granger causality
    print("  [D] Granger causality")
    gr_pid = granger_causality_per_pid(
        alertness_series, composite_series, max_lag,
        x_name="alertness", y_name="composite",
    )
    gr_sum = pd.DataFrame()
    if not gr_pid.empty:
        gr_pid.to_csv(
            OUTPUT_DIR / f"multimodal_{slug}_granger_per_pid.csv", index=False)
        gr_sum = granger_summary(gr_pid)
        gr_sum.to_csv(
            OUTPUT_DIR / f"multimodal_{slug}_granger_summary.csv", index=False)

    return {
        "ccf_perm": perm_df,
        "event_comp_to_alert": s_c2a,
        "event_alert_to_comp": s_a2c,
        "dl": dl,
        "granger_pid": gr_pid,
        "granger_summary": gr_sum,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_multimodal_dynamics(
    max_lag: int = MAX_LAG,
    n_perm: int = N_PERMUTATIONS,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    print(f"Loaded data: shape {df.shape}")

    if ORDER_COL not in df.columns:
        raise ValueError(f"Order column '{ORDER_COL}' not found.")

    df = attach_alertness_score(df)
    print(f"Alertness score: {ALERTNESS_SCORE_COL} = TLX + sleepiness")

    # Build composites for each group
    composite_cols: Dict[str, str] = {}
    for group_name, features in ANALYSIS_GROUPS.items():
        slug = group_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        col = f"__composite_{slug}__"
        composite_cols[group_name] = col
        features_present = [f for f in features if f in df.columns]
        print(f"\n[{group_name}] {len(features_present)}/{len(features)} features available — "
              f"building composite '{col}'")
        df = build_group_composite(df, GROUP_COL, features, col)

    # Run analyses per group
    all_results: Dict[str, Dict] = {}
    for group_name, composite_col in composite_cols.items():
        all_results[group_name] = run_group_analysis(
            df, group_name, composite_col, max_lag, n_perm,
        )

    # Combined comparison figures
    print("\n[Plots] Generating comparison figures …")

    plot_ccf_comparison(
        all_results,
        OUTPUT_DIR / "multimodal_ccf_comparison.png",
    )
    plot_event_comparison(
        all_results, "composite_to_alertness",
        OUTPUT_DIR / "multimodal_event_proxy_to_alertness_comparison.png",
    )
    plot_event_comparison(
        all_results, "alertness_to_composite",
        OUTPUT_DIR / "multimodal_event_alertness_to_proxy_comparison.png",
    )
    plot_dl_comparison(
        all_results,
        OUTPUT_DIR / "multimodal_dl_comparison.png",
    )
    plot_granger_comparison(
        all_results,
        OUTPUT_DIR / "multimodal_granger_comparison.png",
    )
    plot_summary_panel(
        all_results,
        OUTPUT_DIR / "multimodal_summary_panel.png",
    )

    # Print headline CCF results for each group
    print("\n" + "=" * 70)
    print("CCF HEADLINE: alertness vs each feature group (lag k>0 → alertness leads)")
    print("=" * 70)
    for group, res in all_results.items():
        perm = res["ccf_perm"]
        if perm.empty:
            print(f"  {group:20s}  no data")
            continue
        n_sig = int((perm["p_perm"] < 0.05).sum()) if "p_perm" in perm.columns else 0
        pos_sig = int(((perm["lag"] > 0) & (perm["p_perm"] < 0.05)).sum()) if "p_perm" in perm.columns else 0
        neg_sig = int(((perm["lag"] < 0) & (perm["p_perm"] < 0.05)).sum()) if "p_perm" in perm.columns else 0
        lag0 = perm[perm["lag"] == 0]
        r0 = float(lag0["mean_r"].iloc[0]) if not lag0.empty else float("nan")
        p0 = float(lag0["p_perm"].iloc[0]) if "p_perm" in lag0.columns and not lag0.empty else float("nan")
        print(
            f"  {group:20s}  n_sig_lags={n_sig:2d}  "
            f"pos_sig={pos_sig}  neg_sig={neg_sig}  "
            f"lag0_r={r0:+.3f}  lag0_p={p0:.3f}"
        )


def main() -> None:
    run_multimodal_dynamics(max_lag=MAX_LAG, n_perm=N_PERMUTATIONS)


if __name__ == "__main__":
    main()
