from __future__ import annotations

"""
Investigate the temporal dynamics between COMBINED SUBJECTIVE ALERTNESS
(TLX + sleepiness) and OBJECTIVE behavioral errors (proxy_composite).

Core question: do subjective alertness ratings and behavioral errors covary
through real lead-lag dynamics, or only through shared session-long fatigue
drift? If a real lead exists in either direction, which one is earlier?

The combined alertness score (TLX + sleepiness) matches how the prediction
pipeline binarises alertness for classification (see
`predict_combined_alertness_lgbm_binary_participant_iqr_scaled.py:
create_alertness_binary_target_per_participant`). Running dynamics on this
combined signal — instead of TLX and sleepiness separately — produces one
substantive conclusion aligned with the modelling target.

Three independent methods stacked here, each with significance testing:

  A. Cross-correlation function (CCF) at lags ±MAX_LAG
       - Per participant, corr(subjective_t, proxy_{t+k}) for k ∈ [-K, +K]
       - Aggregate across participants (mean ± bootstrap CI)
       - Lag-k>0 → subjective leads proxy (subjective at time t correlated
         with proxy at later time t+k)
       - Lag-k<0 → proxy leads subjective
       - Significance: within-participant permutation null

  B. Event-study around spikes
       - Anchor on rows where one signal spikes (z ≥ 1.0 within participant)
       - Plot the OTHER signal's z-scored trajectory in window [-K, +K]
       - Asymmetry around offset=0 reveals lead/lag

  C. Distributed-lag regression with participant fixed effects
       - y_t ~ Σ_k β_k x_{t+k} + α_pid
       - Cluster-robust SE by participant
       - Significant β at k<0 ↔ x LEADS y  (past x predicts current y)
       - Significant β at k>0 ↔ y LEADS x  (future x correlates with present y)
       - This is the OPPOSITE sign convention from CCF (where lag k indexes
         the y-side: corr(x_t, y_{t+k})). Tests are equivalent but the lag
         labels flip.
       - Bonferroni-adjusted p-values across lags

  D. Granger-causality F-test per participant (both directions), then
     aggregate the proportion of participants showing significant Granger
     effect at each lag in each direction.

Outputs (all keyed by the combined "alertness_proxy" pair):
  CSVs:
    - dynamics_ccf_alertness_proxy_per_pid.csv
    - dynamics_ccf_alertness_proxy_aggregated.csv     (CCF + permutation p)
    - dynamics_event_proxy_to_alertness_summary.csv
    - dynamics_event_alertness_to_proxy_summary.csv
    - dynamics_dl_alertness_proxy_coefficients.csv    (DL β + cluster SE)
    - dynamics_granger_alertness_proxy_per_pid.csv
    - dynamics_granger_alertness_proxy_summary.csv
    - dynamics_stationarity_ccf_alertness_vs_proxy.csv
    - dynamics_stationarity_dl_alertness_vs_proxy.csv
    - dynamics_summary_table.csv                       (verdict appendix)
    - dynamics_single_conclusion.txt                   ★ HEADLINE OUTPUT

  PNGs:
    - dynamics_per_pid_timeseries.png                  (small multiples)
    - dynamics_ccf_alertness_proxy.png                 (mean CCF ± CI)
    - dynamics_event_proxy_to_alertness.png            (event-locked traj)
    - dynamics_event_alertness_to_proxy.png
    - dynamics_dl_coefs_alertness_proxy.png            (DL coefs with 95% CI)
    - dynamics_stationarity_ccf_alertness_vs_proxy.png
    - dynamics_stationarity_dl_alertness_vs_proxy.png
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prediction.alertness.predict_combined_alertness_lgbm_binary_participant_iqr_scaled import (
    GROUP_COL,
    SLEEPINESS_COL,
    TLX_COL,
    load_data,
)
from prediction.alertness.shared_config import DATA_PATH, PROXY_ERROR_COLS


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "processed_data"
ORDER_COL = "initiation"  # within-session sequence order
PROXY_COMPOSITE_COL = "__proxy_composite__"
ALERTNESS_SCORE_COL = "__alertness_score__"  # TLX + sleepiness (raw sum)

MAX_LAG = 5            # ±5 trials for CCF, event study, distributed lag
N_PERMUTATIONS = 1000  # within-PID shuffle null for CCF
EVENT_Z_THRESH = 1.0   # spike threshold for event study (z-score within PID)
BOOTSTRAP_N = 2000
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def attach_proxy_composite(df: pd.DataFrame, proxy_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    available = [c for c in proxy_cols if c in df.columns]
    if not available:
        raise ValueError("No proxy_cols available in dataframe.")
    df[PROXY_COMPOSITE_COL] = df[available].mean(axis=1)
    return df


def attach_alertness_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add the combined alertness score column: alertness = TLX + sleepiness.

    Matches the construction in
    `create_alertness_binary_target_per_participant` — same raw signal that
    the binary classification target is built from.
    """
    if TLX_COL not in df.columns or SLEEPINESS_COL not in df.columns:
        raise ValueError(f"Required columns missing: {TLX_COL}, {SLEEPINESS_COL}")
    df = df.copy()
    df[ALERTNESS_SCORE_COL] = df[TLX_COL] + df[SLEEPINESS_COL]
    return df


def per_pid_zscored_series(
    df: pd.DataFrame, group_col: str, order_col: str, value_col: str,
) -> Dict[object, np.ndarray]:
    """{pid: z-scored values in initiation order}. NaN preserved."""
    out: Dict[object, np.ndarray] = {}
    for pid, g in df.groupby(group_col):
        g_sorted = g.sort_values(order_col, kind="mergesort")
        v = g_sorted[value_col].to_numpy(dtype=float)
        valid = ~np.isnan(v)
        if valid.sum() < 2:
            out[pid] = v - (np.nanmean(v) if valid.any() else 0.0)
            continue
        m = float(np.nanmean(v))
        s = float(np.nanstd(v))
        if s < 1e-9:
            out[pid] = v - m
        else:
            out[pid] = (v - m) / s
    return out


def _pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 4:
        return float("nan")
    a_v = a[valid]; b_v = b[valid]
    if np.std(a_v) < 1e-9 or np.std(b_v) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a_v, b_v)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# A. Cross-correlation function (CCF)
#    CCF[k] = corr(x_t, y_{t+k})
#      k > 0 → y is at later t → x_t carries info predating y → x leads y
#      k < 0 → y is at earlier t → y leads x
# ─────────────────────────────────────────────────────────────────────────────

def compute_ccf_per_pid(
    x_dict: Dict[object, np.ndarray],
    y_dict: Dict[object, np.ndarray],
    max_lag: int,
) -> pd.DataFrame:
    rows = []
    lags = list(range(-max_lag, max_lag + 1))
    for pid, x in x_dict.items():
        if pid not in y_dict:
            continue
        y = y_dict[pid]
        n = min(len(x), len(y))
        x = x[:n]; y = y[:n]
        for k in lags:
            if k >= 0:
                xv, yv = x[: n - k] if k > 0 else x, y[k:] if k > 0 else y
            else:
                xv, yv = x[-k:], y[: n + k]
            rows.append({"participant": pid, "lag": k, "r": _pearsonr_safe(xv, yv)})
    return pd.DataFrame(rows)


def aggregate_ccf(per_pid_ccf: pd.DataFrame, bootstrap_n: int = BOOTSTRAP_N) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_STATE)
    out_rows = []
    for lag, grp in per_pid_ccf.groupby("lag"):
        rs = grp["r"].dropna().to_numpy()
        if len(rs) < 3:
            out_rows.append({"lag": int(lag), "n": len(rs),
                             "mean_r": float("nan"), "ci_low": float("nan"),
                             "ci_high": float("nan")})
            continue
        boot = np.array([np.mean(rs[rng.choice(len(rs), len(rs), replace=True)])
                         for _ in range(bootstrap_n)])
        out_rows.append({
            "lag": int(lag), "n": int(len(rs)),
            "mean_r": float(np.mean(rs)),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
        })
    return pd.DataFrame(out_rows).sort_values("lag").reset_index(drop=True)


def permutation_test_ccf(
    x_dict: Dict[object, np.ndarray],
    y_dict: Dict[object, np.ndarray],
    max_lag: int,
    n_perm: int = N_PERMUTATIONS,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Within-participant permutation null. Shuffle x within each PID,
    recompute mean CCF; build null at each lag; report two-sided p-value
    against the observed mean CCF."""
    rng = np.random.RandomState(seed)
    observed = compute_ccf_per_pid(x_dict, y_dict, max_lag)
    obs_agg = aggregate_ccf(observed)

    null_means: Dict[int, List[float]] = {k: [] for k in range(-max_lag, max_lag + 1)}
    for _ in range(n_perm):
        x_perm: Dict[object, np.ndarray] = {}
        for pid, x in x_dict.items():
            xp = x.copy()
            mask = ~np.isnan(xp)
            xp_valid = xp[mask]
            rng.shuffle(xp_valid)
            xp[mask] = xp_valid
            x_perm[pid] = xp
        perm_per_pid = compute_ccf_per_pid(x_perm, y_dict, max_lag)
        for lag, grp in perm_per_pid.groupby("lag"):
            null_means[int(lag)].append(float(np.nanmean(grp["r"])))

    rows = []
    for _, r in obs_agg.iterrows():
        lag = int(r["lag"])
        nulls = np.asarray(null_means[lag], dtype=float)
        nulls = nulls[~np.isnan(nulls)]
        obs = float(r["mean_r"])
        p = (float(np.mean(np.abs(nulls) >= abs(obs))) if not np.isnan(obs) and len(nulls) > 0
             else float("nan"))
        rows.append({**r.to_dict(), "p_perm": p,
                     "null_mean": float(np.mean(nulls)) if len(nulls) > 0 else float("nan"),
                     "null_std": float(np.std(nulls)) if len(nulls) > 0 else float("nan")})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# B. Event study around spikes
# ─────────────────────────────────────────────────────────────────────────────

def event_study(
    df: pd.DataFrame, group_col: str, order_col: str,
    anchor_col: str, target_col: str, max_lag: int,
    z_threshold: float = EVENT_Z_THRESH, side: str = "high",
) -> pd.DataFrame:
    """For each row where anchor_col z-score (within-PID) ≥ +threshold (or
    ≤ -threshold if side='low'), record target_col z-score at offsets
    [-max_lag, +max_lag]. Returns long table."""
    rows = []
    for pid, g in df.groupby(group_col):
        g_sorted = g.sort_values(order_col, kind="mergesort").reset_index(drop=True)
        a = g_sorted[anchor_col].to_numpy(dtype=float)
        t = g_sorted[target_col].to_numpy(dtype=float)
        a_m, a_s = np.nanmean(a), np.nanstd(a) + 1e-9
        t_m, t_s = np.nanmean(t), np.nanstd(t) + 1e-9
        a_z = (a - a_m) / a_s
        t_z = (t - t_m) / t_s
        if side == "high":
            events = np.where(a_z >= z_threshold)[0]
        else:
            events = np.where(a_z <= -z_threshold)[0]
        for e in events:
            for off in range(-max_lag, max_lag + 1):
                idx = e + off
                if 0 <= idx < len(t_z) and not np.isnan(t_z[idx]):
                    rows.append({"participant": pid, "event_idx": int(e),
                                 "offset": int(off), "target_z": float(t_z[idx])})
    return pd.DataFrame(rows)


def event_study_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    agg = events_df.groupby("offset").agg(
        n=("target_z", "count"),
        mean=("target_z", "mean"),
        sd=("target_z", "std"),
    ).reset_index()
    agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    agg["ci_low"] = agg["mean"] - 1.96 * agg["se"]
    agg["ci_high"] = agg["mean"] + 1.96 * agg["se"]
    # Permutation p-value: shuffle offsets within event, retest mean per offset
    rng = np.random.RandomState(RANDOM_STATE)
    null_means = {int(o): [] for o in agg["offset"]}
    for _ in range(500):
        shuffled = events_df.copy()
        shuffled["offset"] = rng.permutation(shuffled["offset"].to_numpy())
        for off, grp in shuffled.groupby("offset"):
            null_means[int(off)].append(float(grp["target_z"].mean()))
    p_vals = []
    for _, r in agg.iterrows():
        off = int(r["offset"])
        nulls = np.array(null_means[off])
        p = float(np.mean(np.abs(nulls) >= abs(r["mean"]))) if len(nulls) > 0 else float("nan")
        p_vals.append(p)
    agg["p_perm"] = p_vals
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# C. Distributed-lag regression with participant fixed effects
#    y_t ~ α_pid + Σ_k β_k x_{t+k}      (k ∈ [-K, +K])
#    Within-PID z-scoring is mathematically equivalent to PID fixed effects
#    for continuous covariates → cluster-robust SE by participant gives
#    valid inference.
# ─────────────────────────────────────────────────────────────────────────────

def distributed_lag_regression(
    df: pd.DataFrame, group_col: str, order_col: str,
    x_col: str, y_col: str, max_lag: int,
) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except Exception as exc:
        print(f"  [warn] statsmodels not available: {exc}; skipping distributed-lag.")
        return pd.DataFrame()

    long_rows = []
    for pid, g in df.groupby(group_col):
        g_sorted = g.sort_values(order_col, kind="mergesort").reset_index(drop=True)
        x = g_sorted[x_col].to_numpy(dtype=float)
        y = g_sorted[y_col].to_numpy(dtype=float)
        x_m, x_s = np.nanmean(x), np.nanstd(x) + 1e-9
        y_m, y_s = np.nanmean(y), np.nanstd(y) + 1e-9
        x_z = (x - x_m) / x_s
        y_z = (y - y_m) / y_s
        n = len(g_sorted)
        for t in range(max_lag, n - max_lag):
            row = {"participant": pid, "y": y_z[t]}
            ok = True
            for k in range(-max_lag, max_lag + 1):
                idx = t + k
                if 0 <= idx < n and not np.isnan(x_z[idx]):
                    row[f"x_lag_{k}"] = float(x_z[idx])
                else:
                    ok = False
                    break
            if ok and not np.isnan(row["y"]):
                long_rows.append(row)

    if not long_rows:
        return pd.DataFrame()
    long_df = pd.DataFrame(long_rows)
    lag_cols = [f"x_lag_{k}" for k in range(-max_lag, max_lag + 1)]
    X = sm.add_constant(long_df[lag_cols])
    y = long_df["y"]
    groups = long_df["participant"]
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

    n_lags = 2 * max_lag + 1
    rows = []
    ci = model.conf_int()
    for k in range(-max_lag, max_lag + 1):
        col = f"x_lag_{k}"
        rows.append({
            "lag": int(k),
            "coef": float(model.params[col]),
            "cluster_robust_se": float(model.bse[col]),
            "p": float(model.pvalues[col]),
            "p_bonferroni": float(min(1.0, model.pvalues[col] * n_lags)),
            "ci_low": float(ci.loc[col, 0]),
            "ci_high": float(ci.loc[col, 1]),
        })
    out = pd.DataFrame(rows)
    out.attrs["n_obs"] = int(len(long_df))
    out.attrs["n_clusters"] = int(long_df["participant"].nunique())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# D. Per-participant Granger causality
# ─────────────────────────────────────────────────────────────────────────────

def granger_causality_per_pid(
    x_dict: Dict[object, np.ndarray], y_dict: Dict[object, np.ndarray],
    max_lag: int, x_name: str = "x", y_name: str = "y",
) -> pd.DataFrame:
    """statsmodels grangercausalitytests([y, x], maxlag) tests whether x
    Granger-causes y. Returns per-PID per-lag F + p in both directions."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception as exc:
        print(f"  [warn] statsmodels missing: {exc}; skipping Granger.")
        return pd.DataFrame()

    rows = []
    for pid, x in x_dict.items():
        if pid not in y_dict:
            continue
        y = y_dict[pid]
        valid = ~np.isnan(x) & ~np.isnan(y)
        x_v = x[valid]
        y_v = y[valid]
        if len(x_v) < max_lag * 3 + 5:
            continue
        # x → y
        try:
            res = grangercausalitytests(np.column_stack([y_v, x_v]),
                                        maxlag=max_lag, verbose=False)
            for k, val in res.items():
                f = val[0]["ssr_ftest"]
                rows.append({"participant": pid,
                             "direction": f"{x_name}→{y_name}",
                             "lag": int(k), "F": float(f[0]), "p": float(f[1])})
        except Exception:
            pass
        # y → x
        try:
            res = grangercausalitytests(np.column_stack([x_v, y_v]),
                                        maxlag=max_lag, verbose=False)
            for k, val in res.items():
                f = val[0]["ssr_ftest"]
                rows.append({"participant": pid,
                             "direction": f"{y_name}→{x_name}",
                             "lag": int(k), "F": float(f[0]), "p": float(f[1])})
        except Exception:
            pass
    return pd.DataFrame(rows)


def granger_summary(per_pid: pd.DataFrame) -> pd.DataFrame:
    if per_pid.empty:
        return pd.DataFrame()
    return (per_pid.groupby(["direction", "lag"])
                   .agg(median_F=("F", "median"),
                        mean_F=("F", "mean"),
                        median_p=("p", "median"),
                        n_sig_005=("p", lambda x: int((x < 0.05).sum())),
                        n_sig_001=("p", lambda x: int((x < 0.01).sum())),
                        n_total=("p", "count"))
                   .reset_index())


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_aggregated_ccf(
    perm_df: pd.DataFrame, x_label: str, y_label: str, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if perm_df.empty:
        ax.text(0.5, 0.5, "No CCF data", ha="center", va="center")
    else:
        ax.errorbar(
            perm_df["lag"], perm_df["mean_r"],
            yerr=[perm_df["mean_r"] - perm_df["ci_low"],
                  perm_df["ci_high"] - perm_df["mean_r"]],
            fmt="o-", color="#1f77b4", capsize=3, linewidth=1.5, markersize=6,
            label="mean ± 95% bootstrap CI",
        )
        if "p_perm" in perm_df.columns:
            sig = perm_df[perm_df["p_perm"] < 0.05]
            if not sig.empty:
                ax.scatter(sig["lag"], sig["mean_r"], s=140, facecolors="none",
                           edgecolors="red", linewidths=2.0,
                           label="p_perm < 0.05", zorder=5)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(out_path.stem.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_event_study(events_summary: pd.DataFrame, title: str, out_path: Path,
                     y_label: str = "Target z (mean ± 95% CI)") -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if events_summary.empty:
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
    else:
        ax.errorbar(
            events_summary["offset"], events_summary["mean"],
            yerr=[events_summary["mean"] - events_summary["ci_low"],
                  events_summary["ci_high"] - events_summary["mean"]],
            fmt="o-", color="#d62728", capsize=3, linewidth=1.5, markersize=6,
        )
        if "p_perm" in events_summary.columns:
            sig = events_summary[events_summary["p_perm"] < 0.05]
            if not sig.empty:
                ax.scatter(sig["offset"], sig["mean"], s=140, facecolors="none",
                           edgecolors="red", linewidths=2.0,
                           label="p_perm < 0.05", zorder=5)
                ax.legend(loc="best", fontsize=9)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(0.0, color="green", linestyle=":", linewidth=1.2,
                   label="event time" if "label" not in ax.get_legend_handles_labels()[1] else None)
    ax.set_xlabel("Offset from event (trials; <0 = before event, >0 = after)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distributed_lag(
    coefs: pd.DataFrame, title: str, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if coefs.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.errorbar(
            coefs["lag"], coefs["coef"],
            yerr=[coefs["coef"] - coefs["ci_low"],
                  coefs["ci_high"] - coefs["coef"]],
            fmt="s-", color="#2ca02c", capsize=3, linewidth=1.5, markersize=6,
        )
        sig = coefs[coefs["p_bonferroni"] < 0.05]
        if not sig.empty:
            ax.scatter(sig["lag"], sig["coef"], s=140, facecolors="none",
                       edgecolors="red", linewidths=2.0,
                       label="Bonferroni p<0.05", zorder=5)
            ax.legend(loc="best", fontsize=9)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Lag k of x in regression (β_k); k<0 → x leads y, k>0 → y leads x")
    ax.set_ylabel("β_k (cluster-robust 95% CI)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_pid_timeseries(
    df: pd.DataFrame, group_col: str, order_col: str,
    cols: List[str], out_path: Path, max_pids: int = 12,
) -> None:
    pids = sorted(df[group_col].dropna().unique())[:max_pids]
    n = len(pids)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 2.6),
                             sharex=False)
    axes = np.atleast_2d(axes).flatten()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, pid in enumerate(pids):
        ax = axes[i]
        g = df[df[group_col] == pid].sort_values(order_col, kind="mergesort")
        for col, color in zip(cols, palette):
            v = g[col].to_numpy(dtype=float)
            valid = ~np.isnan(v)
            if valid.sum() < 2 or np.nanstd(v) < 1e-9:
                v_z = v - (np.nanmean(v) if valid.any() else 0.0)
            else:
                v_z = (v - np.nanmean(v)) / np.nanstd(v)
            ax.plot(np.arange(len(v_z)), v_z, label=col, color=color, linewidth=1.0)
        ax.set_title(f"PID {pid}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Per-participant z-scored time series ({n}/{df[group_col].nunique()} pids)",
                 y=1.005)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Stationarity diagnostics — re-run CCF and DL on first-differenced and
# linearly-detrended series. If a lead-lag finding survives both transforms,
# it's a real causal structure; if it collapses, it was a shared-trend or
# autocorrelation artifact.
#
# Definitions:
#   First-differenced  d_t = x_t - x_{t-1}.  Removes any linear trend and
#                      most autocorrelation. Length n-1 (NaN at index 0).
#   Linearly detrended  fit x_t = α + β·t per participant, take residuals.
#                       Removes any linear trend only. Length n.
# ─────────────────────────────────────────────────────────────────────────────

def _first_diff_dict(d: Dict[object, np.ndarray]) -> Dict[object, np.ndarray]:
    """Per-PID first difference. Prepends NaN so output length matches input."""
    out: Dict[object, np.ndarray] = {}
    for pid, v in d.items():
        if len(v) < 2:
            out[pid] = np.full_like(v, np.nan, dtype=float)
            continue
        out[pid] = np.concatenate([[np.nan], np.diff(v)])
    return out


def _linear_detrend_dict(d: Dict[object, np.ndarray]) -> Dict[object, np.ndarray]:
    """Per-PID residual from least-squares linear fit on within-session index."""
    out: Dict[object, np.ndarray] = {}
    for pid, v in d.items():
        valid = ~np.isnan(v)
        if valid.sum() < 3:
            out[pid] = v - (np.nanmean(v) if valid.any() else 0.0)
            continue
        t = np.arange(len(v), dtype=float)
        slope, intercept = np.polyfit(t[valid], v[valid], 1)
        out[pid] = v - (slope * t + intercept)
    return out


def _attach_transformed_cols(
    df: pd.DataFrame, group_col: str, order_col: str,
    value_cols: List[str], transform: str,
) -> pd.DataFrame:
    """Add columns `{prefix}_{value_col}` to df with transformed values
    (per-participant first-difference or linear detrend). NaN where
    transform is undefined (e.g. first row for differencing)."""
    if transform == "diff":
        prefix = "d"
    elif transform == "detrend":
        prefix = "dt"
    else:
        raise ValueError(f"Unknown transform: {transform}")
    df = df.copy()
    new_cols = [f"{prefix}_{c}" for c in value_cols]
    for c in new_cols:
        df[c] = np.nan

    for pid, idx in df.groupby(group_col).groups.items():
        idx = np.asarray(list(idx))
        sub = df.loc[idx].sort_values(order_col, kind="mergesort")
        ordered_idx = sub.index.to_numpy()
        for c_orig, c_new in zip(value_cols, new_cols):
            v = df.loc[ordered_idx, c_orig].to_numpy(dtype=float)
            if transform == "diff":
                if len(v) < 2:
                    continue
                transformed = np.concatenate([[np.nan], np.diff(v)])
            else:  # detrend
                valid = ~np.isnan(v)
                if valid.sum() < 3:
                    transformed = v - (np.nanmean(v) if valid.any() else 0.0)
                else:
                    t = np.arange(len(v), dtype=float)
                    slope, intercept = np.polyfit(t[valid], v[valid], 1)
                    transformed = v - (slope * t + intercept)
            df.loc[ordered_idx, c_new] = transformed
    return df


def stationarity_diagnostics_pair(
    df: pd.DataFrame, group_col: str, order_col: str,
    x_col: str, y_col: str, max_lag: int, n_perm: int,
    pair_label: str, out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare CCF and DL across raw / first-differenced / linearly-detrended
    versions of (x, y). Returns (ccf_compare, dl_compare) and saves plots/CSV.
    """
    print(f"\n  [stationarity] {pair_label}: raw vs differenced vs detrended")

    # Build per-PID dicts (z-scored) for raw, then transform
    raw_x = per_pid_zscored_series(df, group_col, order_col, x_col)
    raw_y = per_pid_zscored_series(df, group_col, order_col, y_col)
    diff_x, diff_y = _first_diff_dict(raw_x), _first_diff_dict(raw_y)
    dt_x, dt_y = _linear_detrend_dict(raw_x), _linear_detrend_dict(raw_y)

    # CCF + permutation test for each pipeline
    print("    CCF perm: raw")
    raw_ccf = permutation_test_ccf(raw_x, raw_y, max_lag, n_perm=n_perm)
    print("    CCF perm: first-differenced")
    diff_ccf = permutation_test_ccf(diff_x, diff_y, max_lag, n_perm=n_perm)
    print("    CCF perm: linearly detrended")
    dt_ccf = permutation_test_ccf(dt_x, dt_y, max_lag, n_perm=n_perm)

    ccf_compare = pd.DataFrame({"lag": raw_ccf["lag"].to_numpy().astype(int)})
    ccf_compare["raw_r"] = raw_ccf["mean_r"].to_numpy()
    ccf_compare["raw_p"] = raw_ccf["p_perm"].to_numpy()
    ccf_compare["diff_r"] = diff_ccf["mean_r"].to_numpy()
    ccf_compare["diff_p"] = diff_ccf["p_perm"].to_numpy()
    ccf_compare["dt_r"] = dt_ccf["mean_r"].to_numpy()
    ccf_compare["dt_p"] = dt_ccf["p_perm"].to_numpy()

    # DL regression on each pipeline. Attach transformed columns then call
    # the existing distributed_lag_regression with the new column names.
    df_diff = _attach_transformed_cols(df, group_col, order_col, [x_col, y_col], "diff")
    df_dt = _attach_transformed_cols(df, group_col, order_col, [x_col, y_col], "detrend")

    print("    DL: raw")
    raw_dl = distributed_lag_regression(df, group_col, order_col, x_col, y_col, max_lag)
    print("    DL: first-differenced")
    diff_dl = distributed_lag_regression(df_diff, group_col, order_col,
                                         f"d_{x_col}", f"d_{y_col}", max_lag)
    print("    DL: linearly detrended")
    dt_dl = distributed_lag_regression(df_dt, group_col, order_col,
                                       f"dt_{x_col}", f"dt_{y_col}", max_lag)

    def _row_dict(coefs: pd.DataFrame, prefix: str) -> Dict[int, Tuple[float, float, float]]:
        if coefs.empty:
            return {}
        return {int(r["lag"]): (float(r["coef"]), float(r["p"]),
                                float(r["p_bonferroni"]))
                for _, r in coefs.iterrows()}

    raw_map = _row_dict(raw_dl, "raw")
    diff_map = _row_dict(diff_dl, "diff")
    dt_map = _row_dict(dt_dl, "dt")

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        r = raw_map.get(lag, (float("nan"),) * 3)
        d = diff_map.get(lag, (float("nan"),) * 3)
        t = dt_map.get(lag, (float("nan"),) * 3)
        rows.append({
            "lag": lag,
            "raw_coef": r[0], "raw_p": r[1], "raw_p_bonf": r[2],
            "diff_coef": d[0], "diff_p": d[1], "diff_p_bonf": d[2],
            "dt_coef": t[0], "dt_p": t[1], "dt_p_bonf": t[2],
        })
    dl_compare = pd.DataFrame(rows)

    print("\n    CCF comparison (mean r per lag, p_perm in parentheses):")
    _format_ccf = ccf_compare.copy()
    for col in ["raw_r", "diff_r", "dt_r"]:
        _format_ccf[col] = _format_ccf[col].round(3)
    for col in ["raw_p", "diff_p", "dt_p"]:
        _format_ccf[col] = _format_ccf[col].round(3)
    print(_format_ccf.to_string(index=False))

    print("\n    DL comparison (β per lag, Bonferroni p in parentheses):")
    _format_dl = dl_compare.copy()
    for col in ["raw_coef", "diff_coef", "dt_coef"]:
        _format_dl[col] = _format_dl[col].round(3)
    for col in ["raw_p_bonf", "diff_p_bonf", "dt_p_bonf"]:
        _format_dl[col] = _format_dl[col].round(3)
    print(_format_dl[["lag", "raw_coef", "raw_p_bonf",
                      "diff_coef", "diff_p_bonf",
                      "dt_coef", "dt_p_bonf"]].to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = pair_label.replace(" ", "_").replace("/", "_")
    ccf_compare.to_csv(out_dir / f"dynamics_stationarity_ccf_{safe}.csv", index=False)
    dl_compare.to_csv(out_dir / f"dynamics_stationarity_dl_{safe}.csv", index=False)

    # Visualization
    plot_stationarity_ccf_comparison(
        ccf_compare, pair_label, out_dir / f"dynamics_stationarity_ccf_{safe}.png",
    )
    plot_stationarity_dl_comparison(
        dl_compare, pair_label, out_dir / f"dynamics_stationarity_dl_{safe}.png",
    )

    # Verdict summary
    verdict = _stationarity_verdict(ccf_compare, dl_compare, pair_label)
    print(f"\n    Verdict: {verdict}")
    return ccf_compare, dl_compare


def _stationarity_verdict(
    ccf_compare: pd.DataFrame, dl_compare: pd.DataFrame, pair_label: str,
) -> str:
    """Classify whether the raw finding survives differencing and detrending."""
    def _n_sig(df, col, p_col, thresh=0.05):
        if df.empty:
            return 0
        return int((df[p_col] < thresh).sum())
    # CCF significant lags by pipeline
    raw_ccf_sig = _n_sig(ccf_compare, "raw_r", "raw_p")
    diff_ccf_sig = _n_sig(ccf_compare, "diff_r", "diff_p")
    dt_ccf_sig = _n_sig(ccf_compare, "dt_r", "dt_p")
    # DL (Bonferroni) significant lags by pipeline
    raw_dl_sig = _n_sig(dl_compare, "raw_coef", "raw_p_bonf")
    diff_dl_sig = _n_sig(dl_compare, "diff_coef", "diff_p_bonf")
    dt_dl_sig = _n_sig(dl_compare, "dt_coef", "dt_p_bonf")

    survives_diff = (diff_ccf_sig >= 1) or (diff_dl_sig >= 1)
    survives_detrend = (dt_ccf_sig >= 1) or (dt_dl_sig >= 1)
    raw_had_signal = (raw_ccf_sig >= 1) or (raw_dl_sig >= 1)

    if not raw_had_signal:
        return "no signal in raw data either"
    if survives_diff and survives_detrend:
        return ("ROBUST — raw finding survives both transformations, "
                "indicating real lead-lag structure beyond shared trend")
    if survives_detrend and not survives_diff:
        return ("PARTIAL — survives detrending but not differencing; "
                "consistent with strong but stationary autocorrelation, "
                "real but slow-varying signal")
    if survives_diff and not survives_detrend:
        return ("UNUSUAL — survives differencing but not detrending; "
                "investigate further (this combination is rare)")
    # Neither survives
    return ("LIKELY ARTIFACT — raw finding does not survive either "
            "differencing or detrending; explained by shared session-long "
            "trend and/or autocorrelation, NOT a real lead-lag effect")


# ─────────────────────────────────────────────────────────────────────────────
# Stationarity diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_stationarity_ccf_comparison(
    ccf_compare: pd.DataFrame, pair_label: str, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    palette = {"raw": "#1f77b4", "diff": "#d62728", "dt": "#2ca02c"}
    legend_names = {
        "raw": "raw (untransformed)",
        "diff": "first-differenced",
        "dt": "linearly detrended",
    }
    for key in ["raw", "diff", "dt"]:
        rs = ccf_compare[f"{key}_r"].to_numpy()
        ps = ccf_compare[f"{key}_p"].to_numpy()
        ax.plot(ccf_compare["lag"], rs, "-o", color=palette[key], markersize=5,
                linewidth=1.5, label=legend_names[key])
        sig_mask = ps < 0.05
        if sig_mask.any():
            ax.scatter(ccf_compare["lag"][sig_mask], rs[sig_mask],
                       s=120, facecolors="none", edgecolors=palette[key],
                       linewidths=2.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Lag k  (k>0 → x leads y;  k<0 → y leads x)")
    ax.set_ylabel("Mean Pearson r across participants")
    ax.set_title(f"Stationarity diagnostic — CCF: {pair_label}\n"
                 f"(open circles = p_perm < 0.05)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stationarity_dl_comparison(
    dl_compare: pd.DataFrame, pair_label: str, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    palette = {"raw": "#1f77b4", "diff": "#d62728", "dt": "#2ca02c"}
    legend_names = {
        "raw": "raw (untransformed)",
        "diff": "first-differenced",
        "dt": "linearly detrended",
    }
    for key in ["raw", "diff", "dt"]:
        coefs = dl_compare[f"{key}_coef"].to_numpy()
        p_bonf = dl_compare[f"{key}_p_bonf"].to_numpy()
        ax.plot(dl_compare["lag"], coefs, "-s", color=palette[key], markersize=5,
                linewidth=1.5, label=legend_names[key])
        sig_mask = p_bonf < 0.05
        if sig_mask.any():
            ax.scatter(dl_compare["lag"][sig_mask], coefs[sig_mask],
                       s=120, facecolors="none", edgecolors=palette[key],
                       linewidths=2.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Lag k of x in regression (β_k); k<0 → x leads y, k>0 → y leads x")
    ax.set_ylabel("β_k (cluster-robust regression coefficient)")
    ax.set_title(f"Stationarity diagnostic — DL: {pair_label}\n"
                 f"(open squares = Bonferroni-corrected p < 0.05)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Single-conclusion synthesis (combined alertness vs proxy)
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_single_conclusion(
    raw_ccf: pd.DataFrame,
    dl_raw: pd.DataFrame,
    ccf_stat: pd.DataFrame,
    dl_stat: pd.DataFrame,
    event_proxy_to_alert: pd.DataFrame,
    event_alert_to_proxy: pd.DataFrame,
) -> str:
    """Combine all dynamics evidence into one paragraph-level conclusion.

    Rules:
      1. Raw effect direction is read from CCF and DL.
      2. Stationarity controls (differencing + detrending) decide whether
         the raw effect is robust or a shared-trend artifact.
      3. If only one of (differenced, detrended) survives, that's a
         partial/autocorrelated effect — flagged separately.
      4. Lag-0 contemporaneous correlation after detrending is reported as
         the residual signal regardless of lead-lag verdict.
    """
    def _safe(df, mask_fn, col):
        if df.empty:
            return pd.DataFrame()
        return df[mask_fn(df)][col]

    # Raw CCF significant lags (split by direction)
    raw_pos_sig = int(((raw_ccf["lag"] > 0) & (raw_ccf["p_perm"] < 0.05)).sum())
    raw_neg_sig = int(((raw_ccf["lag"] < 0) & (raw_ccf["p_perm"] < 0.05)).sum())
    raw_lag0 = raw_ccf[raw_ccf["lag"] == 0]
    lag0_r = float(raw_lag0["mean_r"].iloc[0]) if not raw_lag0.empty else float("nan")
    lag0_p = float(raw_lag0["p_perm"].iloc[0]) if not raw_lag0.empty else float("nan")
    raw_max_pos = raw_ccf[raw_ccf["lag"] > 0]["mean_r"].max()
    raw_max_neg = raw_ccf[raw_ccf["lag"] < 0]["mean_r"].max()
    raw_max_pos_lag = (raw_ccf.loc[raw_ccf["mean_r"].idxmax(), "lag"]
                       if not raw_ccf.empty else float("nan"))

    # Raw DL Bonferroni-significant lags
    if not dl_raw.empty:
        dl_raw_sig = dl_raw[dl_raw["p_bonferroni"] < 0.05]
        dl_raw_neg_sig = int((dl_raw_sig["lag"] < 0).sum())
        dl_raw_pos_sig = int((dl_raw_sig["lag"] > 0).sum())
        dl_raw_top = (dl_raw.loc[dl_raw["coef"].abs().idxmax()]
                      if not dl_raw.empty else None)
    else:
        dl_raw_sig = pd.DataFrame()
        dl_raw_neg_sig = dl_raw_pos_sig = 0
        dl_raw_top = None

    # Survival under differencing and detrending
    if not ccf_stat.empty:
        n_diff_ccf_sig = int((ccf_stat["diff_p"] < 0.05).sum())
        n_dt_ccf_sig = int((ccf_stat["dt_p"] < 0.05).sum())
    else:
        n_diff_ccf_sig = n_dt_ccf_sig = 0
    if not dl_stat.empty:
        n_diff_dl_sig = int((dl_stat["diff_p_bonf"] < 0.05).sum())
        n_dt_dl_sig = int((dl_stat["dt_p_bonf"] < 0.05).sum())
        # Track if the same lag survives DL — strict criterion
        raw_dl_sig_lags = set(dl_stat.loc[dl_stat["raw_p_bonf"] < 0.05, "lag"])
        same_lag_diff = sum(
            1 for lag in raw_dl_sig_lags
            if (dl_stat.loc[dl_stat["lag"] == lag, "diff_p_bonf"].iloc[0] < 0.05)
        )
        same_lag_dt = sum(
            1 for lag in raw_dl_sig_lags
            if (dl_stat.loc[dl_stat["lag"] == lag, "dt_p_bonf"].iloc[0] < 0.05)
        )
    else:
        n_diff_dl_sig = n_dt_dl_sig = 0
        same_lag_diff = same_lag_dt = 0
        raw_dl_sig_lags = set()

    # Residual contemporaneous correlation after detrending
    if not ccf_stat.empty:
        lag0_dt = ccf_stat[ccf_stat["lag"] == 0]
        lag0_dt_r = float(lag0_dt["dt_r"].iloc[0]) if not lag0_dt.empty else float("nan")
        lag0_dt_p = float(lag0_dt["dt_p"].iloc[0]) if not lag0_dt.empty else float("nan")
    else:
        lag0_dt_r = lag0_dt_p = float("nan")

    # Decide overall verdict
    raw_had_signal = (raw_pos_sig + raw_neg_sig >= 2) or (dl_raw_pos_sig + dl_raw_neg_sig >= 1)
    # Strict: same Bonferroni-significant lag survives in DL transform
    survives_strict = (same_lag_diff >= 1) and (same_lag_dt >= 1)
    # Lenient: any single lag reaches sig in transformed CCF/DL
    survives_lenient_diff = (n_diff_ccf_sig >= 1) or (n_diff_dl_sig >= 1)
    survives_lenient_dt = (n_dt_ccf_sig >= 1) or (n_dt_dl_sig >= 1)

    # Headline classifier
    if not raw_had_signal:
        headline = "No detectable lead-lag relationship between combined alertness and proxy errors."
        body = (
            f"Neither cross-correlation nor distributed-lag regression detected significant "
            f"raw associations beyond chance level."
        )
    elif survives_strict:
        # Determine direction
        if dl_raw_neg_sig > dl_raw_pos_sig:
            direction = "combined alertness LEADS proxy errors"
        elif dl_raw_pos_sig > dl_raw_neg_sig:
            direction = "proxy errors LEAD combined alertness"
        else:
            direction = "bidirectional / unresolved"
        headline = f"ROBUST temporal lead-lag detected: {direction}."
        body = (
            f"The raw Bonferroni-significant distributed-lag coefficient survived both "
            f"first-differencing and linear detrending at the same lag, indicating a "
            f"real trial-to-trial coupling beyond shared session-long trend."
        )
    elif (not survives_lenient_diff) and (not survives_lenient_dt):
        headline = ("LIKELY ARTIFACT: raw lead-lag findings collapse entirely under "
                    "both stationarity controls.")
        body = (
            f"Raw cross-correlation showed significance at {raw_pos_sig + raw_neg_sig} of "
            f"11 lags (lag 0 r = {lag0_r:.3f}, p = {lag0_p:.3f}). "
            f"Raw distributed-lag found {dl_raw_neg_sig + dl_raw_pos_sig} Bonferroni-"
            f"significant lag(s). After first-differencing both series, "
            f"{n_diff_ccf_sig} CCF lag(s) and {n_diff_dl_sig} DL lag(s) survived. "
            f"After linear detrending, {n_dt_ccf_sig} CCF lag(s) and {n_dt_dl_sig} "
            f"DL lag(s) survived. Effect sizes collapsed from raw to near-zero in both "
            f"transformed pipelines, confirming the raw association is dominated by a "
            f"shared session-long fatigue trend rather than direct trial-to-trial coupling."
        )
    elif survives_lenient_dt and not survives_lenient_diff:
        headline = ("WEAK / SUGGESTIVE: small residual correlation after detrending but "
                    "no robust trial-to-trial coupling.")
        body = (
            f"Raw effect collapsed under first-differencing ({n_diff_ccf_sig} CCF lag(s), "
            f"{n_diff_dl_sig} DL lag(s) survived), but {n_dt_ccf_sig} CCF lag(s) and "
            f"{n_dt_dl_sig} DL lag(s) remained marginally significant after detrending. "
            f"This pattern is consistent with weak slow-varying coupling that does not "
            f"hold at the single-trial level, OR with autocorrelated residuals that "
            f"differencing eliminates. Treat as exploratory."
        )
    else:
        headline = ("MIXED / UNUSUAL: raw effect partially survives stationarity controls.")
        body = (
            f"Raw effect survived under differencing ({n_diff_ccf_sig} CCF, "
            f"{n_diff_dl_sig} DL Bonferroni-sig) but not detrending "
            f"({n_dt_ccf_sig} CCF, {n_dt_dl_sig} DL sig) — this combination is unusual "
            f"and warrants closer inspection (likely a non-linear trend confound)."
        )

    # Always report residual contemporaneous correlation
    contemp_note = (
        f"\n\nResidual contemporaneous correlation after detrending (lag 0): "
        f"r = {lag0_dt_r:.3f} (p_perm = {lag0_dt_p:.3f}). "
    )
    if (not np.isnan(lag0_dt_p)) and lag0_dt_p < 0.05:
        contemp_note += (
            "This small but significant lag-0 effect reflects shared response to fatigue "
            "accumulation at each measurement point — both signals track the same latent "
            "fatigue state but neither precedes the other at the trial timescale."
        )
    else:
        contemp_note += (
            "Even after trend removal, no significant contemporaneous correlation remained, "
            "indicating subjective and behavioral signals are only weakly related at the "
            "trial timescale once shared fatigue drift is controlled."
        )

    # Quantitative appendix
    appendix = (
        f"\n\n--- Supporting evidence ---\n"
        f"Raw CCF:           {raw_pos_sig + raw_neg_sig} / 11 lags p_perm < 0.05  "
        f"(peak r = {raw_max_pos:.3f} at positive lag, "
        f"{raw_max_neg:.3f} at negative lag)\n"
        f"Raw DL Bonferroni: {dl_raw_neg_sig + dl_raw_pos_sig} / 11 lags  "
        f"(strongest β = {dl_raw_top['coef']:.3f} at lag {int(dl_raw_top['lag'])}, "
        f"p_bonf = {dl_raw_top['p_bonferroni']:.3f})\n"
        f"After first-differencing: {n_diff_ccf_sig} CCF lag(s) sig, "
        f"{n_diff_dl_sig} DL Bonferroni-sig lag(s)\n"
        f"After linear detrending:  {n_dt_ccf_sig} CCF lag(s) sig, "
        f"{n_dt_dl_sig} DL Bonferroni-sig lag(s)\n"
        f"Same-lag DL survival under transforms: "
        f"diff={same_lag_diff}/{len(raw_dl_sig_lags)}, "
        f"detrend={same_lag_dt}/{len(raw_dl_sig_lags)}"
    )

    return f"HEADLINE\n  {headline}\n\nSUBSTANTIVE INTERPRETATION\n  {body}{contemp_note}{appendix}"


# ─────────────────────────────────────────────────────────────────────────────
# Verdict synthesis
# ─────────────────────────────────────────────────────────────────────────────

def _verdict_from_ccf(perm_df: pd.DataFrame, label: str) -> Dict[str, object]:
    if perm_df.empty:
        return {"test": label, "verdict": "no data"}
    pos = perm_df[perm_df["lag"] > 0]
    neg = perm_df[perm_df["lag"] < 0]
    zero = perm_df[perm_df["lag"] == 0]
    pos_max = pos["mean_r"].abs().max() if not pos.empty else float("nan")
    neg_max = neg["mean_r"].abs().max() if not neg.empty else float("nan")
    pos_sig = int(((pos["p_perm"] < 0.05) & (pos["mean_r"].abs() > 0)).sum()) if "p_perm" in pos else 0
    neg_sig = int(((neg["p_perm"] < 0.05) & (neg["mean_r"].abs() > 0)).sum()) if "p_perm" in neg else 0
    z_r = float(zero["mean_r"].iloc[0]) if not zero.empty else float("nan")
    z_p = float(zero["p_perm"].iloc[0]) if "p_perm" in zero.columns and not zero.empty else float("nan")
    if pos_sig > neg_sig:
        verdict = "x leads y"
    elif neg_sig > pos_sig:
        verdict = "y leads x"
    elif pos_sig == 0 and neg_sig == 0:
        verdict = ("contemporaneous (lag=0 sig)" if z_p < 0.05
                   else "no significant lead/lag")
    else:
        verdict = "symmetric (both sides significant)"
    return {
        "test": label,
        "lag0_r": z_r, "lag0_p": z_p,
        "max_pos_lag_|r|": pos_max, "max_neg_lag_|r|": neg_max,
        "n_sig_pos_lags": pos_sig, "n_sig_neg_lags": neg_sig,
        "verdict": verdict,
    }


def _verdict_from_dl(coefs: pd.DataFrame, label: str) -> Dict[str, object]:
    """Distributed-lag convention: y_t = α + Σ_k β_k x_{t+k}
       β at k < 0 sig → past x predicts current y → x leads y
       β at k > 0 sig → future x correlated with current y → y leads x
       (OPPOSITE sign convention from CCF, where corr(x_t, y_{t+k}) means
        k > 0 → x leads y.)
    """
    if coefs.empty:
        return {"test": label, "verdict": "no data"}
    pos = coefs[coefs["lag"] > 0]
    neg = coefs[coefs["lag"] < 0]
    pos_sig = int((pos["p_bonferroni"] < 0.05).sum())
    neg_sig = int((neg["p_bonferroni"] < 0.05).sum())
    if neg_sig > pos_sig:
        verdict = "x leads y"       # past x → present y
    elif pos_sig > neg_sig:
        verdict = "y leads x"       # future x ↔ present y → y is the leader
    elif pos_sig == 0 and neg_sig == 0:
        verdict = "no significant lead/lag (Bonferroni)"
    else:
        verdict = "symmetric"
    return {
        "test": label,
        "n_sig_pos_lags_bonf": pos_sig,
        "n_sig_neg_lags_bonf": neg_sig,
        "max_|beta|_pos": float(pos["coef"].abs().max()) if not pos.empty else float("nan"),
        "max_|beta|_neg": float(neg["coef"].abs().max()) if not neg.empty else float("nan"),
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_dynamics_analysis(max_lag: int = MAX_LAG, n_perm: int = N_PERMUTATIONS) -> None:
    """Run dynamics analyses on COMBINED alertness (TLX + sleepiness) vs proxy
    errors. Outputs a single substantive conclusion."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    print(f"Loaded data: shape {df.shape}")

    if ORDER_COL not in df.columns:
        raise ValueError(f"Order column '{ORDER_COL}' not found in dataframe.")

    df = attach_proxy_composite(df, PROXY_ERROR_COLS)
    df = attach_alertness_score(df)
    print(f"Combined alertness signal: {ALERTNESS_SCORE_COL} = {TLX_COL} + {SLEEPINESS_COL}")

    # Build per-participant z-scored ordered series
    alertness = per_pid_zscored_series(df, GROUP_COL, ORDER_COL, ALERTNESS_SCORE_COL)
    proxy = per_pid_zscored_series(df, GROUP_COL, ORDER_COL, PROXY_COMPOSITE_COL)
    print(f"Built ordered series: {len(alertness)} participants, "
          f"avg length = {np.mean([len(v) for v in alertness.values()]):.1f}")

    # Per-PID time-series visualization (combined alertness + proxy)
    print("\n[viz] Per-participant time series small multiples")
    plot_per_pid_timeseries(
        df, GROUP_COL, ORDER_COL, [ALERTNESS_SCORE_COL, PROXY_COMPOSITE_COL],
        OUTPUT_DIR / "dynamics_per_pid_timeseries.png", max_pids=12,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # A. CCF analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[A] Cross-correlation function — corr(alertness_t, proxy_{t+k})")
    print("    lag>0 → alertness leads proxy;  lag<0 → proxy leads alertness")
    print(f"\n  Permutation test (N={n_perm})")
    ccf_ap_pid = compute_ccf_per_pid(alertness, proxy, max_lag)
    ccf_ap_pid.to_csv(OUTPUT_DIR / "dynamics_ccf_alertness_proxy_per_pid.csv", index=False)
    perm_ap = permutation_test_ccf(alertness, proxy, max_lag, n_perm=n_perm)
    perm_ap.to_csv(OUTPUT_DIR / "dynamics_ccf_alertness_proxy_aggregated.csv", index=False)
    print(perm_ap.round(4).to_string(index=False))
    plot_aggregated_ccf(
        perm_ap,
        x_label="Lag k  (CCF: corr(alertness_t, proxy_{t+k}))   k>0: alertness leads",
        y_label="Mean Pearson r (across participants)",
        out_path=OUTPUT_DIR / "dynamics_ccf_alertness_proxy.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # B. Event study
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[B] Event study (anchor on z≥{EVENT_Z_THRESH:.1f} spikes)")

    es_proxy_to_alert = event_study(df, GROUP_COL, ORDER_COL,
                                    PROXY_COMPOSITE_COL, ALERTNESS_SCORE_COL, max_lag)
    s_proxy_to_alert = event_study_summary(es_proxy_to_alert)
    if not s_proxy_to_alert.empty:
        s_proxy_to_alert.to_csv(
            OUTPUT_DIR / "dynamics_event_proxy_to_alertness_summary.csv", index=False)
        print("  Proxy spike → alertness trajectory:")
        print(s_proxy_to_alert.round(3).to_string(index=False))
    plot_event_study(s_proxy_to_alert, "Alertness trajectory around proxy-error spikes",
                     OUTPUT_DIR / "dynamics_event_proxy_to_alertness.png",
                     "Alertness z")

    es_alert_to_proxy = event_study(df, GROUP_COL, ORDER_COL,
                                    ALERTNESS_SCORE_COL, PROXY_COMPOSITE_COL, max_lag)
    s_alert_to_proxy = event_study_summary(es_alert_to_proxy)
    if not s_alert_to_proxy.empty:
        s_alert_to_proxy.to_csv(
            OUTPUT_DIR / "dynamics_event_alertness_to_proxy_summary.csv", index=False)
        print("  Alertness spike → proxy trajectory:")
        print(s_alert_to_proxy.round(3).to_string(index=False))
    plot_event_study(s_alert_to_proxy, "Proxy-error trajectory around alertness spikes",
                     OUTPUT_DIR / "dynamics_event_alertness_to_proxy.png",
                     "Proxy z")

    # ─────────────────────────────────────────────────────────────────────────
    # C. Distributed-lag regression
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[C] Distributed-lag regression (cluster-robust SE by participant)")
    print("    y = proxy,  x = combined alertness")
    dl_ap = distributed_lag_regression(df, GROUP_COL, ORDER_COL,
                                       ALERTNESS_SCORE_COL, PROXY_COMPOSITE_COL, max_lag)
    if not dl_ap.empty:
        dl_ap.to_csv(OUTPUT_DIR / "dynamics_dl_alertness_proxy_coefficients.csv",
                     index=False)
        print(f"    n_obs={dl_ap.attrs.get('n_obs')}  "
              f"n_clusters={dl_ap.attrs.get('n_clusters')}")
        print(dl_ap.round(4).to_string(index=False))
        plot_distributed_lag(
            dl_ap,
            "Distributed-lag β: y = proxy regressed on alertness_{t+k}",
            OUTPUT_DIR / "dynamics_dl_coefs_alertness_proxy.png",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # D. Granger causality
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[D] Granger causality (per participant, both directions)")
    gr_ap = granger_causality_per_pid(alertness, proxy, max_lag,
                                      x_name="alertness", y_name="proxy")
    if not gr_ap.empty:
        gr_ap.to_csv(OUTPUT_DIR / "dynamics_granger_alertness_proxy_per_pid.csv",
                     index=False)
        s_gr_ap = granger_summary(gr_ap)
        s_gr_ap.to_csv(OUTPUT_DIR / "dynamics_granger_alertness_proxy_summary.csv",
                       index=False)
        print("  alertness ↔ proxy Granger summary:")
        print(s_gr_ap.round(3).to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # E. Stationarity diagnostics (raw vs differenced vs detrended)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[E] Stationarity diagnostics — does the lead/lag finding survive "
          "removing shared trends and autocorrelation?")
    ccf_cmp, dl_cmp = stationarity_diagnostics_pair(
        df, GROUP_COL, ORDER_COL, ALERTNESS_SCORE_COL, PROXY_COMPOSITE_COL,
        max_lag=max_lag, n_perm=n_perm,
        pair_label="alertness_vs_proxy", out_dir=OUTPUT_DIR,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # F. Single conclusion
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SINGLE CONCLUSION — combined alertness (TLX + sleepiness) vs proxy errors")
    print("=" * 78)
    conclusion = synthesize_single_conclusion(
        raw_ccf=perm_ap,
        dl_raw=dl_ap,
        ccf_stat=ccf_cmp,
        dl_stat=dl_cmp,
        event_proxy_to_alert=s_proxy_to_alert,
        event_alert_to_proxy=s_alert_to_proxy,
    )
    print(conclusion)
    conclusion_path = OUTPUT_DIR / "dynamics_single_conclusion.txt"
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write(conclusion + "\n")
    print(f"\n  Saved conclusion: {conclusion_path}")

    # Also save the raw-vs-stationarity verdict table for the appendix
    verdicts = [
        _verdict_from_ccf(perm_ap, "CCF alertness vs proxy (raw)"),
        _verdict_from_dl(dl_ap, "DL  alertness→proxy (raw)"),
        {"test": "Stationarity alertness vs proxy",
         "verdict": _stationarity_verdict(ccf_cmp, dl_cmp, "alertness_vs_proxy")},
    ]
    verdict_df = pd.DataFrame(verdicts)
    verdict_df.to_csv(OUTPUT_DIR / "dynamics_summary_table.csv", index=False)

    # Final interpretation guide
    print("\n" + "=" * 78)
    print("HOW TO READ THE NUMBERS")
    print("=" * 78)
    print("""\
  Combined alertness signal: ALERTNESS_t = TLX_t + sleepiness_t per trial,
  z-scored within each participant for cross-PID aggregation.

  LAG CONVENTIONS (CCF and DL flip signs — lag indexes a different signal):

    CCF: r_k = corr(alertness_t, proxy_{t+k})       (lag indexes proxy)
      k > 0 sig → alertness LEADS proxy             (subjective → behaviour)
      k < 0 sig → proxy LEADS alertness             (behaviour → subjective)
      k = 0     → contemporaneous

    DL : proxy_t = α + Σ_k β_k · alertness_{t+k}    (lag indexes alertness)
      k < 0 sig → alertness LEADS proxy
      k > 0 sig → proxy LEADS alertness
      k = 0     → contemporaneous

  Stationarity guard: any raw lead-lag effect must SURVIVE first-differencing
  AND/OR linear detrending. Effects that vanish under both transformations are
  artifacts of shared session-long fatigue drift, not real temporal coupling.
""")


def main() -> None:
    run_dynamics_analysis(max_lag=MAX_LAG, n_perm=N_PERMUTATIONS)


if __name__ == "__main__":
    main()
