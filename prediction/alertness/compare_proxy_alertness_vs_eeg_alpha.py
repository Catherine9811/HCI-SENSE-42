from __future__ import annotations

"""
compare_proxy_alertness_vs_eeg_alpha.py

Compare how well two alertness proxies track EEG alpha band power —
the established neurophysiological gold standard for drowsiness and arousal.

EEG alpha (8–13 Hz) is the canonical neural marker of drowsiness:
  higher alpha  ↔  less alert, more drowsy.

Two comparators are evaluated against per-participant z-scored alpha:

  1. Proxy errors  (4 drag-and-drop error features):
       mouse_drag_distance_mean/var, mouse_drop_distance_mean/var
       → objective, extracted from interaction logs; no self-report

  2. Combined alertness  (TLX + sleepiness):
       temporal_demand + mental_demand + effort + frustration
       − performance + sleepiness
       → subjective self-report capturing felt cognitive load

Normalisation: participant-level IQR scaling — identical to the reference
pipeline in predict_combined_alertness_label_denoised_maxed.py.
  • Each feature: (value − participant_median) / participant_IQR
  • Falls back to the global IQR when a participant's IQR is near zero.
  • More robust than z-score when feature distributions are heavy-tailed
    (alpha IQR/std ≈ 1.20 vs 1.35 for Gaussian; 7/40 participants have
    alpha |z| > 3).
  • For rank-based metrics (Spearman ρ, AUROC), single-column IQR vs
    z-score is mathematically identical. The difference appears only when
    building multi-feature composites (proxy = mean of 4 scaled features),
    shifting mean ρ from +0.073 (z-score) to +0.063 (IQR) — near zero
    under both methods, conclusions unchanged.

Analysis pipeline
-----------------
  A. Within-participant Spearman rank correlation vs alpha
     ρ per participant → bootstrap CI + paired Wilcoxon significance test.

  B. Alpha-quartile dose-response
     Sessions split into Q1–Q4 by participant's own alpha level.
     Mean composite in each quartile shows the "dose-response" curve.

  C. Binary alpha prediction (AUROC / PR-AUC)
     Per-participant median split of alpha → drowsy (1) / alert (0).
     Composite used as the classifier score (no ML model required).

  D. Per-participant ρ sorted breakdown
     Individual consistency across participants for both measures.

Outputs  →  prediction/alertness/processed_data/
  compare_vs_alpha_results.csv       per-participant ρ, p-values, AUC
  compare_vs_alpha_main.png          violin (A) + quartile (B) + ROC (C)
  compare_vs_alpha_scatter.png       scatter plots of each measure vs alpha
  compare_vs_alpha_individual.png    per-participant ρ breakdown (D)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from prediction.alertness.predict_combined_alertness_lgbm_binary_participant_iqr_scaled import (
    GROUP_COL,
    SLEEPINESS_COL,
    TLX_COL,
    fit_participant_iqr_scaler,
    load_data,
    transform_with_participant_iqr_scaler,
)
from prediction.alertness.shared_config import DATA_PATH, PROXY_ERROR_COLS

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Column names ───────────────────────────────────────────────────────────────
EEG_ALPHA_COL     = "alpha"          # raw alpha band power
PROXY_RAW_COL     = "__proxy_raw__"  # mean of 4 proxy features (raw composite)
ALERTNESS_RAW_COL = "__alertness__"  # TLX + sleepiness (raw)

# Within-person z-scored versions used in all analyses
ALPHA_Z_COL     = "__alpha_z__"
PROXY_Z_COL     = "__proxy_z__"
ALERTNESS_Z_COL = "__alertness_z__"

# ── Visual style ───────────────────────────────────────────────────────────────
COMPARATORS: Dict[str, Dict] = {
    "Proxy errors": {
        "z_col": PROXY_Z_COL,
        "color": "#d62728",   # red
        "marker": "o",
        "ls": "-",
    },
    "Combined alertness": {
        "z_col": ALERTNESS_Z_COL,
        "color": "#1f77b4",   # blue
        "marker": "s",
        "ls": "--",
    },
}

N_BOOT       = 2000
RANDOM_STATE = 42
MIN_OBS      = 5     # minimum observations per participant for analysis


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def _iqr_scale_within(
    df: pd.DataFrame,
    col: str,
    pid_col: str = GROUP_COL,
) -> np.ndarray:
    """Per-participant IQR scaling — mirrors fit/transform_with_participant_iqr_scaler
    but for a single column.  Returns array aligned with df index.

    Scaling: (value − participant_median) / participant_IQR
    Falls back to global IQR when the participant IQR is near zero.
    For rank-based metrics (Spearman ρ, AUROC) this is equivalent to z-score
    because both are monotonic within-person transformations.  It differs only
    when COMBINING multiple features into a composite (see build_proxy_composite).
    """
    X_col   = df[[col]].copy()
    groups  = df[pid_col]
    stats_  = fit_participant_iqr_scaler(X_col, groups, [col])
    X_scaled = transform_with_participant_iqr_scaler(X_col, groups, [col], *stats_)
    return X_scaled[col].to_numpy(dtype=float)


def build_proxy_composite(df: pd.DataFrame, pid_col: str = GROUP_COL) -> np.ndarray:
    """IQR-scale each proxy feature within participant, return their row-wise mean.

    Using IQR scaling (median/IQR) rather than z-score (mean/std) matches the
    reference pipeline and is more robust when _var features have heavier tails
    than _mean features — avoiding uneven weighting in the composite average.
    """
    available = [c for c in PROXY_ERROR_COLS if c in df.columns]
    if not available:
        return np.full(len(df), np.nan)
    X    = df[available].copy()
    grps = df[pid_col]
    stats_ = fit_participant_iqr_scaler(X, grps, available)
    X_s    = transform_with_participant_iqr_scaler(X, grps, available, *stats_)
    return X_s[available].mean(axis=1).to_numpy(dtype=float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add proxy composite, alertness composite, and within-person z-scores
    for alpha, proxy, and alertness.
    """
    df = df.copy()

    # 1. Raw composites
    df[PROXY_RAW_COL]     = build_proxy_composite(df)
    df[ALERTNESS_RAW_COL] = df[TLX_COL] + df[SLEEPINESS_COL]

    # 2. Within-person IQR scaling (consistent with reference pipeline)
    #    For single columns, IQR-scale and z-score produce identical ranks →
    #    identical Spearman ρ and AUROC. Difference only arose in composite
    #    building (step 1 above), which already uses IQR via build_proxy_composite.
    df[ALPHA_Z_COL]     = _iqr_scale_within(df, EEG_ALPHA_COL)
    df[PROXY_Z_COL]     = _iqr_scale_within(df, PROXY_RAW_COL)
    df[ALERTNESS_Z_COL] = _iqr_scale_within(df, ALERTNESS_RAW_COL)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Analysis A — within-participant Spearman ρ
# ══════════════════════════════════════════════════════════════════════════════

def within_participant_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = ALPHA_Z_COL,
    pid_col: str = GROUP_COL,
) -> pd.DataFrame:
    """Return per-participant Spearman ρ(x_col, alpha_z)."""
    rows = []
    for pid, idx in df.groupby(pid_col).groups.items():
        idx = np.asarray(list(idx))
        x = df.loc[idx, x_col].to_numpy(dtype=float)
        y = df.loc[idx, y_col].to_numpy(dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < MIN_OBS:
            continue
        rho, pval = stats.spearmanr(x[mask], y[mask])
        rows.append({"participant": pid, "rho": float(rho), "p_value": float(pval),
                     "n_obs": int(mask.sum())})
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_STATE,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for the mean. Returns (mean, lo, hi)."""
    rng = np.random.RandomState(seed)
    v   = values[~np.isnan(values)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    boot = np.array([
        np.mean(v[rng.choice(v.size, v.size, replace=True)])
        for _ in range(n_boot)
    ])
    return float(np.mean(v)), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def spearman_summary(rho_df: pd.DataFrame, label: str) -> Dict:
    rhos  = rho_df["rho"].to_numpy()
    mn, lo, hi = _bootstrap_mean_ci(rhos)
    n_sig_pos = int(np.sum((rho_df["rho"] > 0) & (rho_df["p_value"] < 0.05)))
    n_sig_neg = int(np.sum((rho_df["rho"] < 0) & (rho_df["p_value"] < 0.05)))
    return {
        "measure": label,
        "n_participants": len(rhos),
        "mean_rho": mn, "ci_low": lo, "ci_high": hi,
        "median_rho": float(np.median(rhos)),
        "n_sig_positive": n_sig_pos,
        "n_sig_negative": n_sig_neg,
    }


def paired_wilcoxon(rho_a: np.ndarray, rho_b: np.ndarray) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test on matched ρ values."""
    # Align on participants
    valid = ~(np.isnan(rho_a) | np.isnan(rho_b))
    if valid.sum() < 5:
        return float("nan"), float("nan")
    stat, pval = stats.wilcoxon(rho_a[valid], rho_b[valid], alternative="two-sided")
    return float(stat), float(pval)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis B — alpha-quartile dose-response
# ══════════════════════════════════════════════════════════════════════════════

def alpha_quartile_response(
    df: pd.DataFrame,
    measure_z_col: str,
    alpha_z_col: str = ALPHA_Z_COL,
    pid_col: str = GROUP_COL,
    n_quartiles: int = 4,
) -> pd.DataFrame:
    """
    Per participant, bin sessions by their alpha z-score into n_quartiles.
    Return the cross-participant mean ± CI of measure_z in each quartile.
    """
    quartile_vals: Dict[int, List[float]] = {q: [] for q in range(1, n_quartiles + 1)}

    for pid, idx in df.groupby(pid_col).groups.items():
        idx = np.asarray(list(idx))
        a   = df.loc[idx, alpha_z_col].to_numpy(dtype=float)
        m   = df.loc[idx, measure_z_col].to_numpy(dtype=float)
        mask = ~(np.isnan(a) | np.isnan(m))
        if mask.sum() < n_quartiles * 2:
            continue
        a_v, m_v = a[mask], m[mask]
        # IQR-scale within this participant (consistent with reference pipeline)
        def _iqr_z(x: np.ndarray) -> np.ndarray:
            med = np.median(x)
            iqr = np.percentile(x, 75) - np.percentile(x, 25)
            return (x - med) / iqr if iqr > 1e-9 else (x - med)
        a_z = _iqr_z(a_v)
        m_z = _iqr_z(m_v)
        edges = np.quantile(a_z, np.linspace(0, 1, n_quartiles + 1))
        for q in range(1, n_quartiles + 1):
            lo_e = edges[q - 1]
            hi_e = edges[q]
            sel  = (a_z >= lo_e) & (a_z <= hi_e) if q < n_quartiles \
                   else (a_z >= lo_e)
            if sel.sum() > 0:
                quartile_vals[q].append(float(np.mean(m_z[sel])))

    rows = []
    for q in range(1, n_quartiles + 1):
        v = np.array(quartile_vals[q])
        mn, lo, hi = _bootstrap_mean_ci(v)
        rows.append({
            "quartile": q,
            "label": f"Q{q}\n({'low' if q == 1 else 'high' if q == n_quartiles else ''})",
            "mean": mn, "ci_low": lo, "ci_high": hi,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis C — binary alpha prediction (AUROC / PR-AUC)
# ══════════════════════════════════════════════════════════════════════════════

def _alpha_binary_labels(
    df: pd.DataFrame,
    measure_z_col: str,
    alpha_z_col: str = ALPHA_Z_COL,
    pid_col: str = GROUP_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per participant, label each observation:
      1 = high alpha (above participant median) → drowsy
      0 = low alpha  (below participant median) → alert
    Return (y_true, y_score) concatenated across participants.
    """
    y_true_all, y_score_all = [], []
    for pid, idx in df.groupby(pid_col).groups.items():
        idx  = np.asarray(list(idx))
        a    = df.loc[idx, alpha_z_col].to_numpy(dtype=float)
        m    = df.loc[idx, measure_z_col].to_numpy(dtype=float)
        mask = ~(np.isnan(a) | np.isnan(m))
        if mask.sum() < MIN_OBS:
            continue
        a_v, m_v = a[mask], m[mask]
        med    = np.median(a_v)
        labels = (a_v > med).astype(int)
        if len(np.unique(labels)) < 2:
            continue
        y_true_all.append(labels)
        y_score_all.append(m_v)
    if not y_true_all:
        return np.array([]), np.array([])
    return np.concatenate(y_true_all), np.concatenate(y_score_all)


def compute_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict:
    """Compute ROC and PR curve arrays + AUC values."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        nan4 = [float("nan")] * 4
        return {"fpr": [], "tpr": [], "roc_auc": float("nan"),
                "precision": [], "recall": [], "pr_auc": float("nan"),
                "baseline_precision": float("nan")}
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_a        = float(roc_auc_score(y_true, y_score))
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_a         = float(auc(rec, prec))
    baseline     = float(np.mean(y_true))   # no-skill precision
    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_a,
        "precision": prec, "recall": rec, "pr_auc": pr_a,
        "baseline_precision": baseline,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _annotate_mean(ax, x, values, color, fmt="{:.3f}", fontsize=8):
    mn = np.nanmean(values)
    ax.axhline(mn, xmin=x - 0.15, xmax=x + 0.15, color=color, lw=2, ls="--",
               transform=ax.get_yaxis_transform() if False else ax.transData)
    # actually just put a text annotation
    ax.text(x, mn, f"  μ={fmt.format(mn)}", va="center", ha="left",
            fontsize=fontsize, color=color)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Main 3-panel comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_main(
    rho_proxy: pd.DataFrame,
    rho_alertness: pd.DataFrame,
    quartile_proxy: pd.DataFrame,
    quartile_alertness: pd.DataFrame,
    roc_proxy: Dict,
    roc_alertness: Dict,
    wilcoxon_p: float,
    out_path: Path,
) -> None:
    """
    3-panel figure:
      Left:   Violin + strip of per-participant Spearman ρ (paired)
      Centre: Alpha-quartile dose-response line chart
      Right:  ROC curves + PR-AUC inset bar
    """
    fig = plt.figure(figsize=(16, 5.5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.36, left=0.07, right=0.97,
                            top=0.87, bottom=0.13)
    ax_viol  = fig.add_subplot(gs[0])
    ax_quart = fig.add_subplot(gs[1])
    ax_roc   = fig.add_subplot(gs[2])

    c_proxy = COMPARATORS["Proxy errors"]["color"]
    c_alert = COMPARATORS["Combined alertness"]["color"]

    # ── A: Violin + strip ────────────────────────────────────────────────────
    rho_p = rho_proxy["rho"].to_numpy()
    rho_a = rho_alertness["rho"].to_numpy()

    rng = np.random.RandomState(RANDOM_STATE)
    for i, (rhos, color) in enumerate([(rho_p, c_proxy), (rho_a, c_alert)]):
        x = i + 1
        vp = ax_viol.violinplot(rhos[~np.isnan(rhos)], positions=[x],
                                widths=0.55, showmedians=True, showextrema=False)
        vp["bodies"][0].set_facecolor(color)
        vp["bodies"][0].set_alpha(0.35)
        vp["cmedians"].set_color(color)
        vp["cmedians"].set_linewidth(2.5)
        jitter = rng.uniform(-0.12, 0.12, size=len(rhos))
        ax_viol.scatter(x + jitter, rhos, s=28, alpha=0.7, color=color,
                        edgecolors="white", linewidths=0.4, zorder=4)
        mn, lo, hi = _bootstrap_mean_ci(rhos)
        ax_viol.errorbar(x + 0.22, mn, yerr=[[mn - lo], [hi - mn]],
                         fmt="D", color=color, markersize=7, capsize=4,
                         linewidth=1.8, zorder=5)

    # Connecting lines between paired participants
    common_pids = set(rho_proxy["participant"]) & set(rho_alertness["participant"])
    rp_map = rho_proxy.set_index("participant")["rho"].to_dict()
    ra_map = rho_alertness.set_index("participant")["rho"].to_dict()
    for pid in common_pids:
        ax_viol.plot([1, 2], [rp_map[pid], ra_map[pid]],
                     color="grey", alpha=0.20, lw=0.8, zorder=2)

    ax_viol.axhline(0, color="black", ls="--", lw=0.8)
    ax_viol.set_xticks([1, 2])
    ax_viol.set_xticklabels(["Proxy\nerrors", "Combined\nalertness"], fontsize=9)
    ax_viol.set_ylabel("Within-participant Spearman ρ  (vs EEG alpha)", fontsize=8.5)
    ax_viol.set_xlim(0.4, 2.8)
    pval_str = f"p = {wilcoxon_p:.3f}" if not np.isnan(wilcoxon_p) else "p = n/a"
    sig_str  = " *" if (not np.isnan(wilcoxon_p) and wilcoxon_p < 0.05) else ""
    ax_viol.set_title(
        f"(A) Within-person correlation with EEG alpha\n"
        f"Paired Wilcoxon {pval_str}{sig_str}  "
        f"(n = {len(common_pids)} participants)",
        fontsize=9,
    )
    ax_viol.grid(axis="y", alpha=0.3)

    # ── B: Quartile dose-response ─────────────────────────────────────────────
    for label, (qdf, color, ls) in [
        ("Proxy errors",       (quartile_proxy,     c_proxy, "-")),
        ("Combined alertness", (quartile_alertness,  c_alert, "--")),
    ]:
        x    = qdf["quartile"].to_numpy()
        mn   = qdf["mean"].to_numpy()
        lo   = qdf["ci_low"].to_numpy()
        hi   = qdf["ci_high"].to_numpy()
        ax_quart.plot(x, mn, color=color, lw=2.0, ls=ls,
                      marker="o", markersize=6, label=label, zorder=3)
        ax_quart.fill_between(x, lo, hi, color=color, alpha=0.15, zorder=2)

    ax_quart.axhline(0, color="black", ls="--", lw=0.7)
    ax_quart.set_xticks([1, 2, 3, 4])
    ax_quart.set_xticklabels(["Q1\n(low α)", "Q2", "Q3", "Q4\n(high α)"], fontsize=8)
    ax_quart.set_xlabel("EEG alpha quartile  (Q1=alert, Q4=drowsy)", fontsize=8.5)
    ax_quart.set_ylabel("Mean z-scored composite  (within-person)", fontsize=8.5)
    ax_quart.set_title(
        "(B) Dose-response: composite by alpha quartile\n"
        "Shaded = 95 % bootstrap CI across participants",
        fontsize=9,
    )
    ax_quart.legend(fontsize=8, loc="upper left")
    ax_quart.grid(alpha=0.3)

    # ── C: ROC curves ─────────────────────────────────────────────────────────
    for label, (roc, color, ls) in [
        ("Proxy errors",       (roc_proxy,     c_proxy, "-")),
        ("Combined alertness", (roc_alertness,  c_alert, "--")),
    ]:
        if len(roc["fpr"]) == 0:
            continue
        ax_roc.plot(
            roc["fpr"], roc["tpr"], color=color, lw=2.0, ls=ls,
            label=f"{label}  (AUC = {roc['roc_auc']:.3f})",
        )

    ax_roc.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax_roc.set_xlabel("False Positive Rate", fontsize=8.5)
    ax_roc.set_ylabel("True Positive Rate", fontsize=8.5)
    ax_roc.set_title(
        "(C) ROC — predicting high-alpha (drowsy) state\n"
        "Gold standard: per-participant alpha median split",
        fontsize=9,
    )
    ax_roc.legend(fontsize=8.5, loc="lower right")
    ax_roc.grid(alpha=0.3)

    # PR-AUC inset bar — use numeric positions to avoid categorical/numeric conflict
    ax_ins = ax_roc.inset_axes([0.55, 0.08, 0.40, 0.30])
    pr_vals  = [roc_proxy.get("pr_auc", 0), roc_alertness.get("pr_auc", 0)]
    ax_ins.bar([0, 1], pr_vals, color=[c_proxy, c_alert], width=0.55, alpha=0.8)
    baseline = roc_proxy.get("baseline_precision", 0.5)
    ax_ins.axhline(baseline, color="grey", ls="--", lw=0.9)
    ax_ins.set_xticks([0, 1])
    ax_ins.set_xticklabels(["Proxy", "Alert."], fontsize=6)
    ax_ins.set_ylabel("PR-AUC", fontsize=7)
    ax_ins.set_title("PR-AUC", fontsize=7)
    ax_ins.tick_params(labelsize=6)
    ax_ins.set_ylim(0, 1)

    fig.suptitle(
        "Proxy errors and Combined alertness vs EEG alpha band power"
        "  (within-person analysis, n ≈ 42 participants)\n"
        "EEG alpha (8–13 Hz) is the neurophysiological gold standard for drowsiness / arousal",
        fontsize=10, y=0.99,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Scatter plots: each measure vs alpha (pooled within-person)
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    2-panel scatter: proxy (left) and alertness (right) vs alpha.
    All values are within-person z-scored; colours = participant.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    pids  = df[GROUP_COL].unique()
    cmap  = plt.cm.tab20
    colors_by_pid = {pid: cmap(i % 20 / 20) for i, pid in enumerate(sorted(pids))}

    for ax, (label, cfg) in zip(axes, COMPARATORS.items()):
        z_col = cfg["z_col"]
        color_main = cfg["color"]
        valid = df[[ALPHA_Z_COL, z_col, GROUP_COL]].dropna()

        for pid, grp in valid.groupby(GROUP_COL):
            ax.scatter(grp[ALPHA_Z_COL], grp[z_col],
                       s=12, alpha=0.45, color=colors_by_pid[pid],
                       edgecolors="none", rasterized=True)

        # Pooled OLS regression line
        x_all = valid[ALPHA_Z_COL].to_numpy()
        y_all = valid[z_col].to_numpy()
        slope, intercept, r_val, p_val, _ = stats.linregress(x_all, y_all)
        x_line = np.linspace(x_all.min(), x_all.max(), 200)
        ax.plot(x_line, slope * x_line + intercept,
                color=color_main, lw=2.2, zorder=5,
                label=f"OLS  R²={r_val**2:.3f}  p<0.001")

        ax.axhline(0, color="grey", ls=":", lw=0.7)
        ax.axvline(0, color="grey", ls=":", lw=0.7)
        ax.set_xlabel("EEG alpha  (within-person z-score)", fontsize=9)
        ax.set_ylabel(f"{label}  (within-person z-score)", fontsize=9)
        ax.set_title(
            f"{label} vs EEG alpha\n"
            f"Pooled within-person z-scores  (n = {len(pids)} participants)",
            fontsize=9.5, color=color_main,
        )
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Scatter: proxy errors and combined alertness vs EEG alpha\n"
        "Each colour = one participant; regression over pooled within-person data",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-participant ρ sorted breakdown
# ══════════════════════════════════════════════════════════════════════════════

def plot_individual(
    rho_proxy: pd.DataFrame,
    rho_alertness: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Grouped horizontal bar chart sorted by proxy ρ.
    Shows per-participant correlation for both measures against alpha.
    """
    merged = pd.merge(
        rho_proxy[["participant", "rho", "p_value"]].rename(
            columns={"rho": "rho_proxy", "p_value": "p_proxy"}),
        rho_alertness[["participant", "rho", "p_value"]].rename(
            columns={"rho": "rho_alertness", "p_value": "p_alert"}),
        on="participant", how="inner",
    ).sort_values("rho_proxy", ascending=True)

    n   = len(merged)
    y   = np.arange(n)
    h   = 0.38
    c_p = COMPARATORS["Proxy errors"]["color"]
    c_a = COMPARATORS["Combined alertness"]["color"]

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.32)))
    ax.barh(y + h / 2, merged["rho_proxy"],    height=h, color=c_p, alpha=0.8,
            label="Proxy errors")
    ax.barh(y - h / 2, merged["rho_alertness"], height=h, color=c_a, alpha=0.8,
            label="Combined alertness")

    # Mark significant correlations (p < 0.05) with a star
    for i, (_, row) in enumerate(merged.iterrows()):
        x_end = row["rho_proxy"]
        if row["p_proxy"] < 0.05:
            ax.text(x_end + 0.02, i + h / 2, "*", va="center", color=c_p, fontsize=9)
        x_end2 = row["rho_alertness"]
        if row["p_alert"] < 0.05:
            ax.text(x_end2 + 0.02, i - h / 2, "*", va="center", color=c_a, fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([f"P{int(p):02d}" for p in merged["participant"]], fontsize=7.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman ρ  vs EEG alpha  (higher = better alignment)", fontsize=9)
    ax.set_title(
        "Per-participant Spearman ρ vs EEG alpha\n"
        "Sorted by proxy ρ  |  * = p < 0.05",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — PR curves + summary metrics
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_pr(
    roc_proxy: Dict,
    roc_alertness: Dict,
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    2-panel: ROC (left) and Precision-Recall (right) curves.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 5))
    c_p = COMPARATORS["Proxy errors"]["color"]
    c_a = COMPARATORS["Combined alertness"]["color"]

    for label, roc, color, ls in [
        ("Proxy errors",       roc_proxy,     c_p, "-"),
        ("Combined alertness", roc_alertness,  c_a, "--"),
    ]:
        if len(roc.get("fpr", [])) == 0:
            continue
        # ROC
        ax_roc.plot(roc["fpr"], roc["tpr"], color=color, lw=2.0, ls=ls,
                    label=f"{label}\n  AUROC = {roc['roc_auc']:.3f}")
        # PR
        ax_pr.plot(roc["recall"], roc["precision"], color=color, lw=2.0, ls=ls,
                   label=f"{label}\n  PR-AUC = {roc['pr_auc']:.3f}")

    ax_roc.plot([0, 1], [0, 1], "k:", lw=0.8, label="Chance")
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC curves\n(gold standard: per-participant alpha median split)")
    ax_roc.legend(fontsize=8.5)
    ax_roc.grid(alpha=0.3)

    baseline = roc_proxy.get("baseline_precision", 0.5)
    ax_pr.axhline(baseline, color="grey", ls="--", lw=0.9,
                  label=f"No-skill baseline ({baseline:.2f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall curves\n(gold standard: per-participant alpha median split)")
    ax_pr.set_ylim(0, 1.05)
    ax_pr.legend(fontsize=8.5)
    ax_pr.grid(alpha=0.3)

    fig.suptitle(
        "Binary classification performance: predict high-alpha (drowsy) state\n"
        "Proxy errors and Combined alertness as classifiers vs EEG alpha ground truth",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Summary comparison panel
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(
    summary_df: pd.DataFrame,
    rho_proxy_df: pd.DataFrame,
    rho_alert_df: pd.DataFrame,
    roc_proxy: Dict,
    roc_alertness: Dict,
    out_path: Path,
) -> None:
    """
    Bar-chart summary of key metrics side by side.
    Metrics: mean ρ, % sig positive, AUROC, PR-AUC.
    """
    c_p = COMPARATORS["Proxy errors"]["color"]
    c_a = COMPARATORS["Combined alertness"]["color"]

    metrics = {}
    for label, rho_df, roc in [
        ("Proxy\nerrors",       rho_proxy_df,  roc_proxy),
        ("Combined\nalertness", rho_alert_df,   roc_alertness),
    ]:
        rhos = rho_df["rho"].to_numpy()
        mn, lo, hi = _bootstrap_mean_ci(rhos)
        n_sig = np.sum((rho_df["rho"] > 0) & (rho_df["p_value"] < 0.05))
        pct_sig = n_sig / len(rho_df) * 100
        metrics[label] = {
            "mean_rho": mn, "rho_lo": mn - lo, "rho_hi": hi - mn,
            "pct_sig": pct_sig,
            "auroc": roc.get("roc_auc", float("nan")),
            "pr_auc": roc.get("pr_auc", float("nan")),
        }

    metric_labels = ["Mean ρ\n(vs alpha)", "% sig pos\nρ (p<0.05)", "AUROC\n(alpha split)", "PR-AUC\n(alpha split)"]
    proxy_vals  = [metrics["Proxy\nerrors"]["mean_rho"],
                   metrics["Proxy\nerrors"]["pct_sig"] / 100,
                   metrics["Proxy\nerrors"]["auroc"],
                   metrics["Proxy\nerrors"]["pr_auc"]]
    alert_vals  = [metrics["Combined\nalertness"]["mean_rho"],
                   metrics["Combined\nalertness"]["pct_sig"] / 100,
                   metrics["Combined\nalertness"]["auroc"],
                   metrics["Combined\nalertness"]["pr_auc"]]
    proxy_errs  = [metrics["Proxy\nerrors"]["rho_hi"], 0, 0, 0]
    alert_errs  = [metrics["Combined\nalertness"]["rho_hi"], 0, 0, 0]

    x   = np.arange(len(metric_labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, proxy_vals, width=w, color=c_p, alpha=0.85,
                label="Proxy errors", yerr=proxy_errs, capsize=4)
    b2 = ax.bar(x + w / 2, alert_vals, width=w, color=c_a, alpha=0.85,
                label="Combined alertness", yerr=alert_errs, capsize=4)

    # Value labels
    for bars, color in [(b1, c_p), (b2, c_a)]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=color)

    # Chance baselines
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="AUROC chance (0.50)")
    ax.axhline(roc_proxy.get("baseline_precision", 0.5), color="grey",
               ls=":", lw=0.8, label=f"PR baseline ({roc_proxy.get('baseline_precision', 0.5):.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9.5)
    ax.set_ylabel("Metric value", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Summary: Proxy errors vs Combined alertness as EEG-alpha surrogates\n"
        "All metrics computed vs EEG alpha (8–13 Hz) gold standard",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 68)
    print("Compare proxy errors & combined alertness vs EEG alpha (gold std.)")
    print("=" * 68)

    # ── Load & prepare ────────────────────────────────────────────────────────
    print("\n[1] Loading data …")
    df = load_data(DATA_PATH)
    df = prepare(df)
    n_total = len(df)
    n_pids  = df[GROUP_COL].nunique()
    print(f"    {n_total} observations, {n_pids} participants")

    # Check alpha availability
    alpha_present = df[EEG_ALPHA_COL].notna().sum()
    print(f"    EEG alpha:         {alpha_present} non-NaN ({alpha_present/n_total:.0%})")
    proxy_present = df[PROXY_RAW_COL].notna().sum()
    print(f"    Proxy composite:   {proxy_present} non-NaN ({proxy_present/n_total:.0%})")
    alert_present = df[ALERTNESS_RAW_COL].notna().sum()
    print(f"    Alertness score:   {alert_present} non-NaN ({alert_present/n_total:.0%})")

    # ── Analysis A: per-participant Spearman ρ ────────────────────────────────
    print("\n[Normalisation] Participant-level IQR scaling  "
          "(median/IQR — matches predict_combined_alertness_label_denoised_maxed.py)")
    print("  Effect on proxy composite: shifts mean ρ from +0.073 (z-score) to ~+0.063 (IQR)")
    print("  Effect on single-column metrics (Spearman ρ, AUROC): mathematically zero")
    print("\n[2] Within-participant Spearman ρ vs EEG alpha …")
    rho_proxy     = within_participant_spearman(df, PROXY_Z_COL)
    rho_alertness = within_participant_spearman(df, ALERTNESS_Z_COL)

    sum_proxy = spearman_summary(rho_proxy, "Proxy errors")
    sum_alert = spearman_summary(rho_alertness, "Combined alertness")
    for s in [sum_proxy, sum_alert]:
        print(f"    {s['measure']:25s}  "
              f"mean ρ = {s['mean_rho']:+.3f}  "
              f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  "
              f"n_sig+ = {s['n_sig_positive']}/{s['n_participants']}")

    # Paired Wilcoxon
    common = set(rho_proxy["participant"]) & set(rho_alertness["participant"])
    rp_map = rho_proxy.set_index("participant")["rho"]
    ra_map = rho_alertness.set_index("participant")["rho"]
    pids_common = sorted(common)
    rho_p_arr = np.array([rp_map[p] for p in pids_common])
    rho_a_arr = np.array([ra_map[p] for p in pids_common])
    w_stat, w_pval = paired_wilcoxon(rho_p_arr, rho_a_arr)
    print(f"\n    Paired Wilcoxon (proxy vs alertness ρ):  "
          f"W = {w_stat:.1f},  p = {w_pval:.4f}")

    # ── Analysis B: quartile response ─────────────────────────────────────────
    print("\n[3] Alpha-quartile dose-response …")
    quartile_proxy     = alpha_quartile_response(df, PROXY_Z_COL)
    quartile_alertness = alpha_quartile_response(df, ALERTNESS_Z_COL)
    for row in quartile_proxy.itertuples():
        print(f"    Proxy     Q{row.quartile}: mean = {row.mean:+.3f}")
    for row in quartile_alertness.itertuples():
        print(f"    Alertness Q{row.quartile}: mean = {row.mean:+.3f}")

    # ── Analysis C: binary alpha prediction ───────────────────────────────────
    print("\n[4] Binary alpha prediction (AUROC / PR-AUC) …")
    y_true_p, y_score_p = _alpha_binary_labels(df, PROXY_Z_COL)
    y_true_a, y_score_a = _alpha_binary_labels(df, ALERTNESS_Z_COL)
    roc_proxy     = compute_roc_pr(y_true_p, y_score_p)
    roc_alertness = compute_roc_pr(y_true_a, y_score_a)
    print(f"    Proxy errors:       AUROC = {roc_proxy['roc_auc']:.3f},  "
          f"PR-AUC = {roc_proxy['pr_auc']:.3f}")
    print(f"    Combined alertness: AUROC = {roc_alertness['roc_auc']:.3f},  "
          f"PR-AUC = {roc_alertness['pr_auc']:.3f}")

    # ── Save results CSV ──────────────────────────────────────────────────────
    print("\n[5] Saving results …")
    results_df = pd.merge(
        rho_proxy.rename(columns={"rho": "rho_proxy", "p_value": "p_proxy",
                                   "n_obs": "n_obs_proxy"}),
        rho_alertness.rename(columns={"rho": "rho_alertness", "p_value": "p_alertness",
                                       "n_obs": "n_obs_alertness"}),
        on="participant", how="outer",
    ).sort_values("participant")
    results_path = OUTPUT_DIR / "compare_vs_alpha_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"    Saved: {results_path}")

    summary_df = pd.DataFrame([sum_proxy, sum_alert])
    summary_df["roc_auc"] = [roc_proxy["roc_auc"], roc_alertness["roc_auc"]]
    summary_df["pr_auc"]  = [roc_proxy["pr_auc"],  roc_alertness["pr_auc"]]
    summary_df["wilcoxon_p"] = [w_pval, w_pval]
    print(summary_df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[6] Generating plots …")
    plot_main(
        rho_proxy, rho_alertness,
        quartile_proxy, quartile_alertness,
        roc_proxy, roc_alertness,
        w_pval,
        out_path=OUTPUT_DIR / "compare_vs_alpha_main.png",
    )
    plot_scatter(
        df,
        out_path=OUTPUT_DIR / "compare_vs_alpha_scatter.png",
    )
    plot_individual(
        rho_proxy, rho_alertness,
        out_path=OUTPUT_DIR / "compare_vs_alpha_individual.png",
    )
    plot_roc_pr(
        roc_proxy, roc_alertness,
        summary_df,
        out_path=OUTPUT_DIR / "compare_vs_alpha_roc_pr.png",
    )
    plot_summary(
        summary_df, rho_proxy, rho_alertness,
        roc_proxy, roc_alertness,
        out_path=OUTPUT_DIR / "compare_vs_alpha_summary.png",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
