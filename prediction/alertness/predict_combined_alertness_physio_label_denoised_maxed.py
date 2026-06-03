from __future__ import annotations

"""
predict_combined_alertness_physio_label_denoised_maxed.py

Identical to predict_combined_alertness_label_denoised_maxed.py except
that the **3rd voter in the multi-modal agreement label** is changed from
the proxy error composite (drag/drop distances) to a **physiological
composite** (EEG alpha + ECG RR intervals + respiratory durations).

Why this variant?
─────────────────
The original proxy voter is objective but behavioural — it may agree
with TLX/sleepiness partly because all three co-vary with task difficulty
rather than purely with internal alertness state.  Replacing it with
neural + autonomic signals provides an orthogonal physiological oracle:
  • EEG alpha power   — cortical arousal marker (higher = more drowsy)
  • Cardiac RR mean/var — autonomic regulation (higher RR = slower HR)
  • Respiratory inhalation/exhalation durations — breathing rate proxy

Key differences from predict_combined_alertness_label_denoised_maxed.py
────────────────────────────────────────────────────────────────────────
  1. `create_physio_composite_binary_target_per_participant()` replaces
     `create_proxy_error_binary_target_per_participant()` as the 3rd voter.
     Steps: (a) IQR-scale each physio feature within participant,
            (b) row-wise mean → composite,
            (c) per-participant lower/upper-percentile split → 0/1/NaN.

  2. PROXY_ERROR_COLS remain **excluded from predictors** (same as before)
     to prevent leakage from the error-proxy columns even though they no
     longer participate in labeling.

  3. Physiological engineered features are also excluded from predictors:
       eng_eeg_theta_over_beta, eng_eeg_alpha_over_beta,
       eng_eeg_theta_alpha_over_beta, eng_eeg_delta_over_beta,
       eng_hrv_ratio, eng_resp_inhale_exhale_ratio.
     The behavioural engineered features (head movement, blink composites)
     are kept.  This ensures no physiological signal leaks into predictors
     via the engineered feature channel.

  4. Output files use the prefix  physio_label_denoised_*.
"""

from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prediction.alertness.predict_combined_alertness_lgbm_binary_participant_iqr_scaled import (
    BINARY_TARGET_COL,
    GROUP_COL,
    RANDOM_STATE,
    SLEEPINESS_COL,
    TLX_COL,
    fit_participant_iqr_scaler,
    load_data,
    prepare_subset_with_target,
    transform_with_participant_iqr_scaler,
)
from prediction.alertness.predict_combined_alertness_ensemble_maxed import (
    ENGINEERED_FEATURE_NAMES,
    add_engineered_features,
)
from prediction.alertness.shared_config import DATA_PATH, FEATURE_GROUPS, PROXY_ERROR_COLS

# ── Reuse unchanged machinery from the original label-denoised script ─────────
from prediction.alertness.predict_combined_alertness_label_denoised_maxed import (
    # smoothing
    attach_smoothed_signals,
    TLX_SMOOTH_COL,
    SLEEP_SMOOTH_COL,
    SCORE_SMOOTH_COL,
    EWMA_ALPHA,
    # label helpers
    _per_pid_binary_signal,
    _make_old_combined_label,
    attach_soft_target,
    SOFT_TARGET_COL,
    WEIGHT_COL,
    OLD_TARGET_COL,
    TLX_SIGNAL_COL,
    SLEEP_SIGNAL_COL,
    # model helpers
    build_preprocessor,
    build_lgbm_classifier,
    build_lgbm_regressor,
    build_clf_pipeline,
    build_reg_pipeline,
    LGBM_PARAMS_CLF,
    LGBM_PARAMS_REG,
    # SHAP selection
    shap_select_on_soft,
    # CV + evaluation
    _confusion,
    _train_oof_heads,
    cross_cv_two_head_stacked,
    within_cv_two_head,
    cross_cv_two_head_stacked_inner_shap,
    validation_old_label_on_same_rows,
    best_threshold_global,
    per_pid_threshold_metrics,
    _compute_pr_auc,
    _print_block,
    _evaluate_predictions_against_label,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
TARGET_GROUP = "mouse_keyboard_traits_sleep_engagement_behavioural"
OUTPUT_DIR = BASE_DIR / "processed_data"

# Per-participant percentile bounds for each modality's high/low label
LOW_PCTL = 0.33
HIGH_PCTL = 0.66

SHAP_IMPORTANCE_PATH    = OUTPUT_DIR / "physio_label_denoised_shap_importance.csv"
SELECTED_FEATURES_PATH  = OUTPUT_DIR / "physio_label_denoised_selected_features.csv"
LABEL_AUDIT_PATH        = OUTPUT_DIR / "physio_label_denoised_label_audit.csv"
VALIDATION_DROPPED_PATH = OUTPUT_DIR / "physio_label_denoised_validation_dropped_rows.csv"
VALIDATION_INNER_SHAP_PATH  = OUTPUT_DIR / "physio_label_denoised_validation_inner_shap.csv"
VALIDATION_OLD_LABEL_PATH   = OUTPUT_DIR / "physio_label_denoised_validation_old_label_same_rows.csv"
VALIDATION_ALL_ROWS_PATH     = OUTPUT_DIR / "physio_label_denoised_validation_lopo_all_rows.csv"
VALIDATION_ALL_ROWS_PROBA_PATH = OUTPUT_DIR / "physio_label_denoised_validation_lopo_all_rows_proba.csv"

# ── Physiological signal columns used as the 3rd voter ────────────────────────
PHYSIO_SIGNAL_COLS: List[str] = [
    "alpha",                                    # EEG alpha power
    "cardiac_rr_interval_mean",                 # ECG: mean RR interval
    "cardiac_rr_interval_var",                  # ECG: RR variability
    "respiratory_inhalation_duration_mean",     # Resp: inhalation duration
    "respiratory_inhalation_duration_var",
    "respiratory_exhalation_duration_mean",     # Resp: exhalation duration
    "respiratory_exhalation_duration_var",
    # "eng_eeg_theta_over_beta",
    # "eng_eeg_alpha_over_beta",
    # "eng_eeg_theta_alpha_over_beta",
    # "eng_eeg_delta_over_beta",
    # "eng_hrv_ratio",
    # "eng_resp_inhale_exhale_ratio",
]

PHYSIO_TARGET_COL  = "__physio_binary__"
PHYSIO_SIGNAL_COL  = "__physio_signal__"

# Physiological engineered features to exclude from predictors.
# These are added by add_engineered_features() but must not be used as
# predictors since physiological signals are now the label oracle.
PHYSIO_ENGINEERED_EXCLUSIONS: set = {
    "eng_eeg_theta_over_beta",
    "eng_eeg_alpha_over_beta",
    "eng_eeg_theta_alpha_over_beta",
    "eng_eeg_delta_over_beta",
    "eng_hrv_ratio",
    "eng_resp_inhale_exhale_ratio",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Physiological composite binary signal
# ═════════════════════════════════════════════════════════════════════════════

def create_physio_composite_binary_target_per_participant(
    df: pd.DataFrame,
    group_col: str,
    physio_cols: List[str],
    lower_percentile: float,
    upper_percentile: float,
    new_col: str,
) -> pd.DataFrame:
    """
    Build a per-participant binary target from the IQR-scaled physiological
    composite:
      1. IQR-scale each available physio feature within participant.
      2. Row-wise mean  →  composite  (higher = drowsier/more fatigued).
      3. Per-participant percentile thresholds:
           composite ≤ lower_pctl → 0  (alert)
           composite ≥ upper_pctl → 1  (fatigued)
           else                   → NaN (middle band)

    Missing physio columns are skipped with a warning; if none are present
    the new column is all-NaN (the physio voter abstains everywhere).
    """
    df = df.copy()
    available = [c for c in physio_cols if c in df.columns]
    missing   = [c for c in physio_cols if c not in df.columns]

    if missing:
        print(f"  [physio signal] Skipping missing columns: {missing}")
    if not available:
        print(f"  [physio signal] No physio columns available — setting {new_col}=NaN")
        df[new_col] = np.nan
        return df

    # IQR-scale within participant
    X      = df[available].copy()
    groups = df[group_col]
    stats_ = fit_participant_iqr_scaler(X, groups, available)
    X_s    = transform_with_participant_iqr_scaler(X, groups, available, *stats_)

    # Row-wise mean composite
    tmp_col = "__physio_composite_tmp__"
    df[tmp_col] = X_s[available].mean(axis=1)

    # Per-participant percentile split
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby(group_col).groups.items():
        idx = np.asarray(list(idx))
        v   = df.loc[idx, tmp_col].to_numpy(dtype=float)
        valid = ~np.isnan(v)
        if valid.sum() == 0:
            continue
        lo = np.quantile(v[valid], lower_percentile)
        hi = np.quantile(v[valid], upper_percentile)
        for i, x in zip(idx, v):
            if np.isnan(x):
                continue
            if x <= lo:
                out[i] = 0.0
            elif x >= hi:
                out[i] = 1.0

    df[new_col] = out
    df.drop(columns=[tmp_col], inplace=True)

    n_high = int(np.nansum(out == 1))
    n_low  = int(np.nansum(out == 0))
    n_nan  = int(np.isnan(out).sum())
    print(f"  [physio signal] {new_col}: high={n_high}, low={n_low}, "
          f"middle/NaN={n_nan}  (from {len(available)} physio features)")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2. Multi-modal agreement labeling (physio replaces proxy)
# ═════════════════════════════════════════════════════════════════════════════

def build_denoised_labels(
    df: pd.DataFrame,
    group_col: str = GROUP_COL,
    low_pctl: float = LOW_PCTL,
    high_pctl: float = HIGH_PCTL,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Three-voter agreement labeling:
      Voter 1: smoothed TLX signal
      Voter 2: smoothed sleepiness signal
      Voter 3: physiological composite signal  ← replaces proxy errors

    Voting:
      label = 1  if n_high ≥ 2 and n_low == 0
      label = 0  if n_low  ≥ 2 and n_high == 0
      else NaN   (ambiguous)

    Confidence weights:
      3/3 agree             → 1.00
      2/3 agree, 1 abstain  → 0.70
      2/3 agree, 1 disagrees→ 0.50
    """
    df = df.copy()

    # Voter 3: physiological composite
    df = create_physio_composite_binary_target_per_participant(
        df=df,
        group_col=group_col,
        physio_cols=PHYSIO_SIGNAL_COLS,
        lower_percentile=low_pctl,
        upper_percentile=high_pctl,
        new_col=PHYSIO_TARGET_COL,
    )

    tlx_sig   = _per_pid_binary_signal(df, group_col, TLX_SMOOTH_COL,  low_pctl, high_pctl)
    sleep_sig = _per_pid_binary_signal(df, group_col, SLEEP_SMOOTH_COL, low_pctl, high_pctl)
    physio_sig = df[PHYSIO_TARGET_COL].to_numpy(dtype=float)

    sigs     = np.stack([tlx_sig, sleep_sig, physio_sig], axis=1)
    n_high   = np.nansum(sigs == 1.0, axis=1)
    n_low    = np.nansum(sigs == 0.0, axis=1)
    n_present = np.sum(~np.isnan(sigs), axis=1)

    label  = np.full(len(df), np.nan)
    weight = np.zeros(len(df), dtype=float)

    for i in range(len(df)):
        h, lo_, present = int(n_high[i]), int(n_low[i]), int(n_present[i])
        if h >= 2 and lo_ == 0:
            label[i] = 1.0
            if h == 3:
                weight[i] = 1.00
            elif present == 3 and lo_ == 1:
                weight[i] = 0.50
            else:
                weight[i] = 0.70
        elif lo_ >= 2 and h == 0:
            label[i] = 0.0
            if lo_ == 3:
                weight[i] = 1.00
            elif present == 3 and h == 1:
                weight[i] = 0.50
            else:
                weight[i] = 0.70

    df[BINARY_TARGET_COL] = label
    df[WEIGHT_COL]        = weight
    df[TLX_SIGNAL_COL]    = tlx_sig
    df[SLEEP_SIGNAL_COL]  = sleep_sig
    df[PHYSIO_SIGNAL_COL] = physio_sig   # store for validation

    buckets = []
    for h_, lo_ in [(3, 0), (2, 0), (2, 1), (0, 3), (0, 2), (1, 2)]:
        n = int(np.sum((n_high == h_) & (n_low == lo_)))
        if n > 0:
            buckets.append({"n_high": h_, "n_low": lo_, "rows": n})
    audit = pd.DataFrame(buckets)
    return df, weight, audit


# ═════════════════════════════════════════════════════════════════════════════
# 3. Validation V1 — predict on dropped rows
#    (proxy signal replaced with physio signal in reference labels)
# ═════════════════════════════════════════════════════════════════════════════

def validation_predict_on_dropped(
    df: pd.DataFrame,
    selected_features: List[str],
    hard_target_col: str,
    soft_target_col: str,
    weight_col: str,
    group_col: str,
    low_pctl: float,
    high_pctl: float,
) -> pd.DataFrame:
    """V1. Train on agreement-labeled rows; predict on dropped rows;
    compare against alternative reference labels."""
    print("\n[V1] Extrapolation to dropped rows")
    df_v1 = _make_old_combined_label(df, group_col, low_pctl, high_pctl, OLD_TARGET_COL)

    cols = list(dict.fromkeys(
        selected_features + [
            hard_target_col, soft_target_col, weight_col, group_col,
            OLD_TARGET_COL, TLX_SIGNAL_COL, SLEEP_SIGNAL_COL, PHYSIO_SIGNAL_COL,
        ]
    ))
    sub = df_v1[cols].copy()
    sub = sub[sub[group_col].notna()].reset_index(drop=True)

    X = sub[selected_features]
    groups = sub[group_col]
    numeric_features = [c for c in selected_features if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in selected_features if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    y_hard  = sub[hard_target_col].to_numpy(dtype=float)
    y_soft  = sub[soft_target_col].to_numpy(dtype=float)
    weight  = sub[weight_col].to_numpy(dtype=float)

    labeled_mask = ~np.isnan(y_hard)
    dropped_mask = ~labeled_mask
    print(f"  Train on {int(labeled_mask.sum())} labelled rows; "
          f"predict on {int(dropped_mask.sum())} dropped rows")

    sw_lab  = np.where(weight[labeled_mask] > 0, weight[labeled_mask], 1.0)
    full_a  = build_clf_pipeline(numeric_features, categorical_features)
    full_a.fit(X_scaled[labeled_mask], y_hard[labeled_mask].astype(int),
               model__sample_weight=sw_lab)

    soft_mask = ~np.isnan(y_soft)
    full_b    = build_reg_pipeline(numeric_features, categorical_features)
    full_b.fit(X_scaled[soft_mask], y_soft[soft_mask])

    proba_a_drop = full_a.predict_proba(X_scaled[dropped_mask])[:, 1]
    score_b_drop = np.clip(full_b.predict(X_scaled[dropped_mask]), 0.0, 1.0)
    proba_drop   = 0.5 * proba_a_drop + 0.5 * score_b_drop

    refs = {
        "old_combined_TLX_plus_sleepiness (33/67)": sub.loc[dropped_mask, OLD_TARGET_COL].to_numpy(),
        "tlx_signal_only (smoothed, 33/67)":         sub.loc[dropped_mask, TLX_SIGNAL_COL].to_numpy(),
        "sleepiness_signal_only (smoothed, 33/67)":  sub.loc[dropped_mask, SLEEP_SIGNAL_COL].to_numpy(),
        "physio_signal_only (33/67)":                sub.loc[dropped_mask, PHYSIO_SIGNAL_COL].to_numpy(),
    }
    rows = [_evaluate_predictions_against_label(proba_drop, ref, name)
            for name, ref in refs.items()]

    proba_a_all = full_a.predict_proba(X_scaled)[:, 1]
    score_b_all = np.clip(full_b.predict(X_scaled), 0.0, 1.0)
    proba_all   = 0.5 * proba_a_all + 0.5 * score_b_all
    for name, ref in {
        "ALL rows vs old_combined (sanity, train+test mixed)": sub[OLD_TARGET_COL].to_numpy(),
        "ALL rows vs agreement label (sanity, includes train)": sub[hard_target_col].to_numpy(),
    }.items():
        rows.append(_evaluate_predictions_against_label(proba_all, ref, name))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. Validation V4 — LOPO on all rows
#    (proxy signal replaced with physio signal in reference labels and pred table)
# ═════════════════════════════════════════════════════════════════════════════

def validation_lopo_all_rows(
    df: pd.DataFrame,
    selected_features: List[str],
    hard_target_col: str,
    soft_target_col: str,
    weight_col: str,
    group_col: str,
    low_pctl: float,
    high_pctl: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """V4. Generate LOPO predictions for ALL rows; evaluate against multiple
    reference labels including the physio signal at 50/50 split."""
    print("\n[V4] LOPO predictions on ALL rows (agreed + disagreed)")

    df_v4 = _make_old_combined_label(df, group_col, low_pctl, high_pctl, OLD_TARGET_COL)
    df_v4 = _make_old_combined_label(df_v4, group_col, 0.50, 0.50, "__old_50_50__")
    df_v4 = df_v4.copy()
    df_v4["__tlx_50_50__"] = _per_pid_binary_signal(
        df_v4, group_col, TLX_SMOOTH_COL, 0.50, 0.50)
    df_v4["__sleep_50_50__"] = _per_pid_binary_signal(
        df_v4, group_col, SLEEP_SMOOTH_COL, 0.50, 0.50)

    # Physio composite at 50/50 split
    df_v4 = create_physio_composite_binary_target_per_participant(
        df=df_v4,
        group_col=group_col,
        physio_cols=PHYSIO_SIGNAL_COLS,
        lower_percentile=0.50,
        upper_percentile=0.50,
        new_col="__physio_50_50__",
    )

    cols = list(dict.fromkeys(
        selected_features + [
            hard_target_col, soft_target_col, weight_col, group_col,
            OLD_TARGET_COL, "__old_50_50__",
            TLX_SIGNAL_COL, SLEEP_SIGNAL_COL, PHYSIO_SIGNAL_COL,
            "__tlx_50_50__", "__sleep_50_50__", "__physio_50_50__",
        ]
    ))
    sub    = df_v4[cols].copy()
    sub    = sub[sub[group_col].notna()].reset_index(drop=True)

    X = sub[selected_features]
    numeric_features     = [c for c in selected_features if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in selected_features if c not in numeric_features]
    groups       = sub[group_col]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled     = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    y_hard  = sub[hard_target_col].to_numpy(dtype=float)
    y_soft  = sub[soft_target_col].to_numpy(dtype=float)
    weight  = sub[weight_col].to_numpy(dtype=float)

    all_proba = np.full(len(sub), np.nan, dtype=float)
    logo      = LeaveOneGroupOut()
    n_groups  = int(groups.nunique())
    print(f"    LOPO over {n_groups} participants, predicting all rows")

    for fold_i, (tr, te) in enumerate(
        logo.split(X_scaled, np.zeros(len(sub)), groups), start=1
    ):
        tr_hard = tr[~np.isnan(y_hard[tr])]
        if len(tr_hard) < 10 or len(np.unique(y_hard[tr_hard])) < 2:
            continue
        sw = np.where(weight[tr_hard] > 0, weight[tr_hard], 1.0)
        pa = build_clf_pipeline(numeric_features, categorical_features)
        pa.fit(X_scaled.iloc[tr_hard], y_hard[tr_hard].astype(int),
               model__sample_weight=sw)
        proba_a = pa.predict_proba(X_scaled.iloc[te])[:, 1]

        tr_soft = tr[~np.isnan(y_soft[tr])]
        if len(tr_soft) >= 10:
            pb = build_reg_pipeline(numeric_features, categorical_features)
            pb.fit(X_scaled.iloc[tr_soft], y_soft[tr_soft])
            score_b = np.clip(pb.predict(X_scaled.iloc[te]), 0.0, 1.0)
            all_proba[te] = 0.5 * proba_a + 0.5 * score_b
        else:
            all_proba[te] = proba_a

        if fold_i % 5 == 0 or fold_i == n_groups:
            print(f"      fold {fold_i}/{n_groups} done")

    coverage = np.mean(~np.isnan(all_proba))
    print(f"    Predictions generated for "
          f"{int(np.sum(~np.isnan(all_proba)))}/{len(sub)} rows ({coverage:.1%})")

    references = [
        ("[agreement] hard label (where available)",
         sub[hard_target_col].to_numpy(dtype=float)),
        (f"[cut-MATCHED {low_pctl:.2f}/{high_pctl:.2f}] OLD combined TLX+sleepiness",
         sub[OLD_TARGET_COL].to_numpy(dtype=float)),
        (f"[cut-MATCHED {low_pctl:.2f}/{high_pctl:.2f}] tlx_signal",
         sub[TLX_SIGNAL_COL].to_numpy(dtype=float)),
        (f"[cut-MATCHED {low_pctl:.2f}/{high_pctl:.2f}] sleep_signal",
         sub[SLEEP_SIGNAL_COL].to_numpy(dtype=float)),
        (f"[cut-MATCHED {low_pctl:.2f}/{high_pctl:.2f}] physio_signal",
         sub[PHYSIO_SIGNAL_COL].to_numpy(dtype=float)),
        ("[UNIVERSAL 50/50] OLD combined TLX+sleepiness",
         sub["__old_50_50__"].to_numpy(dtype=float)),
        ("[UNIVERSAL 50/50] tlx_signal",
         sub["__tlx_50_50__"].to_numpy(dtype=float)),
        ("[UNIVERSAL 50/50] sleep_signal",
         sub["__sleep_50_50__"].to_numpy(dtype=float)),
        ("[UNIVERSAL 50/50] physio_signal",
         sub["__physio_50_50__"].to_numpy(dtype=float)),
    ]
    rows = [_evaluate_predictions_against_label(all_proba, ref, name)
            for name, ref in references]

    agreed_mask    = ~np.isnan(y_hard)
    disagreed_mask = np.isnan(y_hard)
    universal_ref  = sub["__old_50_50__"].to_numpy(dtype=float)
    for subset_name, m in [
        ("[UNIVERSAL 50/50] AGREED rows only (vs OLD combined)",    agreed_mask),
        ("[UNIVERSAL 50/50] DISAGREED rows only (vs OLD combined)", disagreed_mask),
        ("[UNIVERSAL 50/50] ALL rows (vs OLD combined)",
         np.ones(len(sub), dtype=bool)),
    ]:
        proba_sub = np.where(m, all_proba, np.nan)
        ref_sub   = np.where(m, universal_ref, np.nan)
        rows.append(_evaluate_predictions_against_label(proba_sub, ref_sub, subset_name))

    eval_df = pd.DataFrame(rows)
    print("\n  V4 results (out-of-fold LOPO predictions on all rows):")
    print("    NB: [cut-MATCHED] excludes middle-band; [UNIVERSAL] comparable across cuts.")
    print(eval_df.round(3).to_string(index=False))

    pred_table = pd.DataFrame({
        "participant":                              groups.to_numpy(),
        "agreement_label":                          y_hard,
        f"old_combined_{low_pctl:.2f}_{high_pctl:.2f}": sub[OLD_TARGET_COL].to_numpy(dtype=float),
        f"tlx_signal_{low_pctl:.2f}_{high_pctl:.2f}":   sub[TLX_SIGNAL_COL].to_numpy(dtype=float),
        f"sleep_signal_{low_pctl:.2f}_{high_pctl:.2f}":  sub[SLEEP_SIGNAL_COL].to_numpy(dtype=float),
        f"physio_signal_{low_pctl:.2f}_{high_pctl:.2f}": sub[PHYSIO_SIGNAL_COL].to_numpy(dtype=float),
        "old_combined_50_50":    sub["__old_50_50__"].to_numpy(dtype=float),
        "tlx_signal_50_50":      sub["__tlx_50_50__"].to_numpy(dtype=float),
        "sleep_signal_50_50":    sub["__sleep_50_50__"].to_numpy(dtype=float),
        "physio_signal_50_50":   sub["__physio_50_50__"].to_numpy(dtype=float),
        "lopo_proba":            all_proba,
    })
    return eval_df, pred_table


# ═════════════════════════════════════════════════════════════════════════════
# 5. Main experiment
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(
    low_pctl: float  = LOW_PCTL,
    high_pctl: float = HIGH_PCTL,
    keep_ratio: float  = 0.0,
    min_features: int  = 25,
    precision_floor: float = 0.50,
    meta_top_k: int    = 5,
    run_v1_dropped: bool    = True,
    run_v2_inner_shap: bool = True,
    run_v3_old_label: bool  = True,
    run_v4_all_rows: bool   = True,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    print(f"Loaded data: shape {df.shape}")

    df = add_engineered_features(df)
    df = attach_smoothed_signals(df, group_col=GROUP_COL)
    print(f"After engineering + smoothing: shape {df.shape}")

    # ── Stage 1: physiological-signal agreement labeling ─────────────────────
    print("\n[Stage 1] Multi-modal agreement labeling  "
          "(3rd voter: physiological composite)")
    df, weights, audit = build_denoised_labels(
        df, group_col=GROUP_COL, low_pctl=low_pctl, high_pctl=high_pctl)
    audit.to_csv(LABEL_AUDIT_PATH, index=False)
    print("  Label audit (per (n_high, n_low) bucket):")
    print(audit.to_string(index=False))
    n_labeled = int(np.sum(~np.isnan(df[BINARY_TARGET_COL])))
    bal = pd.Series(df[BINARY_TARGET_COL]).value_counts(normalize=True).sort_index()
    print(f"  Total labelled rows: {n_labeled} / {len(df)}  (balance: {bal.to_dict()})")
    n_3of3        = int(np.sum(df[WEIGHT_COL] == 1.0))
    n_2of3_abstain  = int(np.sum(df[WEIGHT_COL] == 0.7))
    n_2of3_disagree = int(np.sum(df[WEIGHT_COL] == 0.5))
    print(f"  Confidence: 3/3 agree = {n_3of3}, 2/3 abstain = {n_2of3_abstain}, "
          f"2/3 disagree = {n_2of3_disagree}")

    df = attach_soft_target(df, group_col=GROUP_COL)

    # ── Feature universe ──────────────────────────────────────────────────────
    #
    # LEAKAGE PREVENTION (two levels):
    #
    # (a) Proxy error columns: excluded even though they are no longer used
    #     as the 3rd voter, to maintain comparability and prevent indirect
    #     leakage through the (now-unused) proxy vote channel.
    #
    # (b) Physiological engineered features: the raw physio columns (alpha,
    #     cardiac_*, respiratory_*) feed the 3rd voter.  Their engineered
    #     derivatives (EEG band ratios, HRV ratio, resp inhale/exhale ratio)
    #     would allow the model to partially reconstruct the physio vote.
    #     Strip them from the predictor set.
    #
    proxy_set  = set(PROXY_ERROR_COLS)
    raw_features_full = FEATURE_GROUPS.get(TARGET_GROUP, [])
    raw_features = [f for f in raw_features_full if f not in proxy_set]

    n_proxy_removed = len(raw_features_full) - len(raw_features)
    print(f"\n  Excluded {n_proxy_removed} proxy-error columns: "
          f"{sorted(proxy_set & set(raw_features_full))}")

    # Engineered features: add all, then filter out physio-derived ones
    all_eng = ENGINEERED_FEATURE_NAMES
    safe_eng = [f for f in all_eng if f not in PHYSIO_ENGINEERED_EXCLUSIONS]
    n_eng_removed = len(all_eng) - len(safe_eng)
    print(f"  Excluded {n_eng_removed} physiological engineered features: "
          f"{sorted(PHYSIO_ENGINEERED_EXCLUSIONS & set(all_eng))}")

    extended_features = [f for f in (raw_features + safe_eng) if f in df.columns]
    print(f"  Extended feature count: {len(extended_features)}")

    # ── Stage 2: SHAP feature selection ───────────────────────────────────────
    print("\n[Stage 2] SHAP feature selection (regressor on soft target)")
    selected, shap_imp = shap_select_on_soft(
        df, extended_features, soft_target_col=SOFT_TARGET_COL,
        group_col=GROUP_COL, keep_ratio=keep_ratio, min_features=min_features,
    )
    shap_imp.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    pd.DataFrame({"selected_feature": selected}).to_csv(SELECTED_FEATURES_PATH, index=False)
    print(f"  Selected {len(selected)} / {len(extended_features)} features:")
    for i, f in enumerate(selected, 1):
        marker = "  [ENG]" if f.startswith("eng_") else ""
        print(f"    {i:2d}. {f}{marker}")

    # ── Stage 3: within-CV ────────────────────────────────────────────────────
    print("\n[Stage 3] Within-participant CV (two-head average: clf+reg)")
    wp_m, wp_yt, wp_ys, wp_gr = within_cv_two_head(
        df, selected, BINARY_TARGET_COL, SOFT_TARGET_COL, WEIGHT_COL, GROUP_COL,
    )
    _print_block("@ threshold 0.50", wp_m)
    wp_pr = _compute_pr_auc(wp_yt, wp_ys)
    print(f"    PR-AUC: {wp_pr:.3f}  baseline: "
          f"{np.mean(wp_yt) if wp_yt.size else float('nan'):.3f}")
    wp_g  = best_threshold_global(wp_yt, wp_ys, precision_floor=precision_floor)
    print(f"  Global threshold-opt: t={wp_g['best_threshold']:.2f}  "
          f"f1={wp_g['f1']:.3f}  precision={wp_g['precision']:.3f}  recall={wp_g['recall']:.3f}")
    wp_pp = per_pid_threshold_metrics(wp_yt, wp_ys, wp_gr)
    _print_block("Per-pid threshold (IN-SAMPLE upper bound — not deployable)", wp_pp)

    # ── Stage 4: cross-CV ─────────────────────────────────────────────────────
    print("\n[Stage 4] Cross-participant CV — two-head stacked LOPO")
    meta_extra = selected[:meta_top_k]
    cp_m, cp_yt, cp_ys = cross_cv_two_head_stacked(
        df, selected, BINARY_TARGET_COL, SOFT_TARGET_COL, WEIGHT_COL, GROUP_COL,
        meta_extra_features=meta_extra,
    )
    _print_block("@ threshold 0.50", cp_m)
    cp_pr = _compute_pr_auc(cp_yt, cp_ys)
    print(f"    PR-AUC: {cp_pr:.3f}  baseline: "
          f"{np.mean(cp_yt) if cp_yt.size else float('nan'):.3f}")
    cp_g = best_threshold_global(cp_yt, cp_ys, precision_floor=precision_floor)
    print(f"  Global threshold-opt: t={cp_g['best_threshold']:.2f}  "
          f"f1={cp_g['f1']:.3f}  precision={cp_g['precision']:.3f}  recall={cp_g['recall']:.3f}")

    # ── Validations ───────────────────────────────────────────────────────────
    v1_table = v2_metrics = v2_pr = v2_g = v3_results = None

    if run_v1_dropped:
        v1_table = validation_predict_on_dropped(
            df, selected, BINARY_TARGET_COL, SOFT_TARGET_COL, WEIGHT_COL, GROUP_COL,
            low_pctl=low_pctl, high_pctl=high_pctl,
        )
        v1_table.to_csv(VALIDATION_DROPPED_PATH, index=False)

    if run_v2_inner_shap:
        v2_metrics, v2_yt, v2_ys, v2_fold_records = cross_cv_two_head_stacked_inner_shap(
            df, extended_features,
            BINARY_TARGET_COL, SOFT_TARGET_COL, WEIGHT_COL, GROUP_COL,
            keep_ratio=keep_ratio, min_features=min_features, meta_top_k=meta_top_k,
        )
        _print_block("V2 cross-CV @ 0.50 (SHAP per fold, leakage-free)", v2_metrics)
        v2_pr = _compute_pr_auc(v2_yt, v2_ys)
        v2_g  = best_threshold_global(v2_yt, v2_ys, precision_floor=precision_floor)
        print(f"      PR-AUC={v2_pr:.3f}  global-t f1={v2_g['f1']:.3f} "
              f"(t={v2_g['best_threshold']:.2f})")
        if not v2_fold_records.empty:
            v2_fold_records.to_csv(VALIDATION_INNER_SHAP_PATH, index=False)

    if run_v3_old_label:
        v3_results = validation_old_label_on_same_rows(
            df, selected, BINARY_TARGET_COL, SOFT_TARGET_COL, WEIGHT_COL, GROUP_COL,
            low_pctl=low_pctl, high_pctl=high_pctl,
            precision_floor=precision_floor, meta_top_k=meta_top_k,
        )
        if v3_results:
            pd.DataFrame([
                {"split": "within", **v3_results.get("within", {})},
                {"split": "cross",  **v3_results.get("cross",  {})},
            ]).to_csv(VALIDATION_OLD_LABEL_PATH, index=False)

    v4_eval = None
    if run_v4_all_rows:
        v4_eval, v4_pred = validation_lopo_all_rows(
            df, selected, BINARY_TARGET_COL, SOFT_TARGET_COL, WEIGHT_COL, GROUP_COL,
            low_pctl=low_pctl, high_pctl=high_pctl,
        )
        v4_eval.to_csv(VALIDATION_ALL_ROWS_PATH,      index=False)
        v4_pred.to_csv(VALIDATION_ALL_ROWS_PROBA_PATH, index=False)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY  (physio-denoised labels + two-head stack)")
    print("3rd voter: EEG alpha + ECG RR + Resp durations  (replaces proxy errors)")
    print("=" * 80)
    print("Headline result (physio-agreement labels, single-pass SHAP):")
    print(f"  Within @ 0.50  :  f1={wp_m.get('f1', float('nan')):.3f}  PR-AUC={wp_pr:.3f}")
    print(f"  Within global  :  f1={wp_g['f1']:.3f}  (t={wp_g['best_threshold']:.2f})")
    print(f"  Within per-pid :  f1={wp_pp.get('f1', float('nan')):.3f}  (in-sample upper bound)")
    print(f"  Cross  @ 0.50  :  f1={cp_m.get('f1', float('nan')):.3f}  PR-AUC={cp_pr:.3f}")
    print(f"  Cross  global  :  f1={cp_g['f1']:.3f}  (t={cp_g['best_threshold']:.2f})")

    if v2_metrics and v2_metrics:
        print("\nV2 — leakage-free (SHAP inside each LOPO fold):")
        print(f"  Cross  @ 0.50  :  f1={v2_metrics.get('f1', float('nan')):.3f}  PR-AUC={v2_pr:.3f}")
        print(f"  Cross  global  :  f1={v2_g['f1']:.3f}  (t={v2_g['best_threshold']:.2f})")
        delta = v2_metrics.get('f1', float('nan')) - cp_m.get('f1', float('nan'))
        print(f"  Δ vs headline  :  {delta:+.3f}")

    if v3_results:
        v3_w = v3_results.get("within", {})
        v3_c = v3_results.get("cross", {})
        print(f"\nV3 — same {v3_results.get('n_intersection', '?')} rows, OLD label scheme:")
        print(f"  Within @ 0.50  :  f1={v3_w.get('f1', float('nan')):.3f}  "
              f"PR-AUC={v3_w.get('pr_auc', float('nan')):.3f}")
        print(f"  Cross  @ 0.50  :  f1={v3_c.get('f1', float('nan')):.3f}  "
              f"PR-AUC={v3_c.get('pr_auc', float('nan')):.3f}")
        dw = v3_w.get('f1', float('nan')) - wp_m.get('f1', float('nan'))
        dc = v3_c.get('f1', float('nan')) - cp_m.get('f1', float('nan'))
        print(f"  Δ within (OLD - physio-agreement): {dw:+.3f}")
        print(f"  Δ cross  (OLD - physio-agreement): {dc:+.3f}")

    if v1_table is not None:
        print(f"\nV1 — predictions on dropped rows:")
        for _, row in v1_table.iterrows():
            n = int(row.get('n', 0))
            if n == 0:
                continue
            print(f"  vs {row['label']}:")
            print(f"    n={n}  f1@0.50={row['f1@0.50']:.3f}  "
                  f"acc={row['accuracy@0.50']:.3f}  kappa={row['kappa@0.50']:.3f}  "
                  f"PR-AUC={row['pr_auc']:.3f}  ROC-AUC={row['roc_auc']:.3f}")

    if v4_eval is not None and not v4_eval.empty:
        print(f"\nV4 — LOPO predictions on ALL rows (out-of-fold; UNIVERSAL refs):")
        focus_names = [
            "[UNIVERSAL 50/50] AGREED rows only (vs OLD combined)",
            "[UNIVERSAL 50/50] DISAGREED rows only (vs OLD combined)",
            "[UNIVERSAL 50/50] ALL rows (vs OLD combined)",
        ]
        for _, row in v4_eval[v4_eval["label"].isin(focus_names)].iterrows():
            print(f"  {row['label']}:")
            print(f"    n={int(row['n'])}  f1@0.50={row['f1@0.50']:.3f}  "
                  f"acc={row['accuracy@0.50']:.3f}  kappa={row['kappa@0.50']:.3f}  "
                  f"PR-AUC={row['pr_auc']:.3f}  ROC-AUC={row['roc_auc']:.3f}  "
                  f"f1_best_t={row['f1_best_t']:.3f}")

    print(f"\n  SHAP importance:        {SHAP_IMPORTANCE_PATH}")
    print(f"  Selected features:      {SELECTED_FEATURES_PATH}")
    print(f"  Label audit:            {LABEL_AUDIT_PATH}")
    if run_v1_dropped:
        print(f"  V1 dropped-row eval:    {VALIDATION_DROPPED_PATH}")
    if run_v2_inner_shap:
        print(f"  V2 inner-SHAP folds:    {VALIDATION_INNER_SHAP_PATH}")
    if run_v3_old_label:
        print(f"  V3 OLD-label results:   {VALIDATION_OLD_LABEL_PATH}")
    if run_v4_all_rows:
        print(f"  V4 all-rows LOPO:       {VALIDATION_ALL_ROWS_PATH}")
        print(f"  V4 per-row predictions: {VALIDATION_ALL_ROWS_PROBA_PATH}")


def main() -> None:
    run_experiment(
        low_pctl=LOW_PCTL,
        high_pctl=HIGH_PCTL,
        keep_ratio=0.0,
        min_features=25,
        precision_floor=0.50,
        meta_top_k=5,
    )


if __name__ == "__main__":
    main()
