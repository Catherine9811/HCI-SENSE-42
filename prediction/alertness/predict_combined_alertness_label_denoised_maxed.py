from __future__ import annotations

"""
Attack the data side: denoise the subjective TLX + sleepiness labels BEFORE
binarization, then train on cleaner targets.

Why this should help (where the previous scripts couldn't):

The performance ceiling we hit (~F1 0.70 cross) is set by **label noise**, not
modeling capacity. TLX + sleepiness are 7-point Likert scales where any single
rating is ±1 noisy. The 35/65 percentile cuts then convert this analog signal
into a hard 0/1 — small Likert wobbles flip labels right at the boundary.

Strategies stacked here:

  1. **Temporal smoothing**: alertness changes slowly within a session. Apply
     a light EWMA (alpha=0.6) to TLX and sleepiness within each participant's
     `initiation` sequence to suppress single-trial rating noise.

  2. **Multi-modal agreement labeling**: build three INDEPENDENT binary
     signals per participant — tlx_binary, sleepiness_binary, error_proxy_binary
     (objective behavioral errors). Hard-label a row only when ≥2 of 3 agree.
     Drops ambiguous rows to NaN. This is much stronger than the old combined
     `tlx + sleepiness` percentile cut because error_proxy adds an OBJECTIVE
     view that the subjective Likert noise cannot corrupt.

  3. **Confidence-weighted training**: 1.0 weight when 3/3 modalities agree,
     0.7 when 2/3 agree (one abstains via "middle band"), 0.5 when one
     modality disagrees outright. Rewards consistent, penalizes confused.

  4. **Soft regression target**: per-participant min-max-scale the smoothed
     alertness score → continuous [0, 1] target. Train an LGBM REGRESSOR on
     this; trees learn smoother decision surfaces than they do on hard 0/1.

  5. **Two-head stacked model**:
       Head A: LGBM CLASSIFIER on agreement-based hard label (weighted)
       Head B: LGBM REGRESSOR on soft (continuous) label
       Meta:   LogReg on [head_A_proba, head_B_score, top SHAP raw features]
     This fuses the binary discriminator with the continuous score — they
     fail in different ways, so combining them helps.

  6. **Engineered features**: carried over from `predict_combined_alertness_ensemble_maxed.py`
     (EEG band ratios, head-movement composites, blink/look-down ratios, etc.).
     SHAP showed they earn 4 of 20 selected slots — keep them.

  7. **Honest evaluation**: report (a) within-CV @ 0.50 and at globally-tuned
     threshold, (b) cross-CV (LOPO) @ 0.50 and at globally-tuned threshold,
     (c) per-participant in-sample threshold ONLY as a labelled upper bound.

What's intentionally NOT here (validated as not helping or out of scope):

  - Multi-seed soft-vote ensembling (LGBM + XGB + CatBoost): the previous
    `_ensemble_maxed.py` showed this regressed cross PR-AUC.
  - Probability calibration (CalibratedClassifierCV): same script showed it
    starves base learners of training data under LOPO.
  - MixUp augmentation: trees benefit less than NNs; not worth the complexity.
  - Hyperparameter search: separate concern, see `optimize_*.py`.
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
from prediction.alertness.predict_proxy_error_lgbm_binary import (
    PROXY_TARGET_COL,
    create_proxy_error_binary_target_per_participant,
)
from prediction.alertness.shared_config import DATA_PATH, FEATURE_GROUPS, PROXY_ERROR_COLS


BASE_DIR = Path(__file__).resolve().parent
TARGET_GROUP = "mouse_keyboard_traits_sleep_engagement_behavioural"
OUTPUT_DIR = BASE_DIR / "processed_data"
SHAP_IMPORTANCE_PATH = OUTPUT_DIR / "label_denoised_maxed_shap_importance.csv"
SELECTED_FEATURES_PATH = OUTPUT_DIR / "label_denoised_maxed_selected_features.csv"
LABEL_AUDIT_PATH = OUTPUT_DIR / "label_denoised_maxed_label_audit.csv"
VALIDATION_DROPPED_PATH = OUTPUT_DIR / "label_denoised_validation_dropped_rows.csv"
VALIDATION_INNER_SHAP_PATH = OUTPUT_DIR / "label_denoised_validation_inner_shap.csv"
VALIDATION_OLD_LABEL_PATH = OUTPUT_DIR / "label_denoised_validation_old_label_same_rows.csv"
VALIDATION_ALL_ROWS_PATH = OUTPUT_DIR / "label_denoised_validation_lopo_all_rows.csv"
VALIDATION_ALL_ROWS_PROBA_PATH = OUTPUT_DIR / "label_denoised_validation_lopo_all_rows_proba.csv"
OLD_TARGET_COL = "__old_combined_target__"
TLX_SIGNAL_COL = "__tlx_signal__"
SLEEP_SIGNAL_COL = "__sleep_signal__"
PROXY_SIGNAL_COL = "__proxy_signal__"

# Smoothing strength: 1.0 = no smoothing (raw); 0.0 = full smoothing.
# 0.6 retains most of the trial-level signal but removes 1-point Likert wobble.
EWMA_ALPHA = 0.6
TLX_SMOOTH_COL = "__tlx_smooth__"
SLEEP_SMOOTH_COL = "__sleep_smooth__"
SCORE_SMOOTH_COL = "__alertness_score_smooth__"
SOFT_TARGET_COL = "__soft_alertness__"
WEIGHT_COL = "__sample_weight__"

# Per-participant percentile bounds for each modality's high/low label
LOW_PCTL = 0.49
HIGH_PCTL = 0.51

LGBM_PARAMS_CLF = {
    "n_estimators": 500,
    "learning_rate": 0.025,
    "num_leaves": 9,
    "min_child_samples": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.0,
    "reg_lambda": 0.05,
}
LGBM_PARAMS_REG = {
    "n_estimators": 500,
    "learning_rate": 0.025,
    "num_leaves": 9,
    "min_child_samples": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.0,
    "reg_lambda": 0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Temporal smoothing per participant
# ─────────────────────────────────────────────────────────────────────────────

def _ewma_within_participant(
    df: pd.DataFrame,
    group_col: str,
    src_col: str,
    out_col: str,
    alpha: float,
    order_col: Optional[str] = "initiation",
) -> pd.DataFrame:
    """Apply per-participant EWMA on `src_col` ordered by `order_col`.
    Falls back to row order if order_col missing."""
    df = df.copy()
    if order_col not in df.columns:
        order_col = None
    smoothed = np.full(len(df), np.nan)
    for _, idx in df.groupby(group_col).groups.items():
        idx = np.asarray(list(idx))
        if order_col is not None:
            sub = df.loc[idx, [src_col, order_col]].copy()
            sub_sorted = sub.sort_values(order_col, kind="mergesort")
            order_idx = sub_sorted.index.to_numpy()
        else:
            order_idx = idx
        vals = df.loc[order_idx, src_col].to_numpy(dtype=float)
        ew = pd.Series(vals).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        smoothed[order_idx] = ew
    df[out_col] = smoothed
    return df


def attach_smoothed_signals(df: pd.DataFrame, group_col: str = GROUP_COL) -> pd.DataFrame:
    """Add EWMA-smoothed TLX, sleepiness, and combined alertness score."""
    df = _ewma_within_participant(df, group_col, TLX_COL, TLX_SMOOTH_COL, EWMA_ALPHA)
    df = _ewma_within_participant(df, group_col, SLEEPINESS_COL, SLEEP_SMOOTH_COL, EWMA_ALPHA)
    df[SCORE_SMOOTH_COL] = df[TLX_SMOOTH_COL] + df[SLEEP_SMOOTH_COL]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-modal agreement labeling
# ─────────────────────────────────────────────────────────────────────────────

def _per_pid_binary_signal(
    df: pd.DataFrame,
    group_col: str,
    score_col: str,
    low_pctl: float,
    high_pctl: float,
) -> np.ndarray:
    """Return per-row signal: 1 (high), 0 (low), or NaN (middle band)."""
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby(group_col).groups.items():
        idx = np.asarray(list(idx))
        vals = df.loc[idx, score_col].to_numpy()
        v = vals[~np.isnan(vals)]
        if v.size == 0:
            continue
        lo = np.quantile(v, low_pctl)
        hi = np.quantile(v, high_pctl)
        for i, x in zip(idx, vals):
            if np.isnan(x):
                continue
            if x <= lo:
                out[i] = 0.0
            elif x >= hi:
                out[i] = 1.0
    return out


def build_denoised_labels(
    df: pd.DataFrame,
    group_col: str = GROUP_COL,
    low_pctl: float = LOW_PCTL,
    high_pctl: float = HIGH_PCTL,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build agreement-based binary labels and confidence weights.

    Three modalities (each gives 0/1/nan per row):
      - tlx_signal: smoothed TLX, per-participant percentiles
      - sleep_signal: smoothed sleepiness, per-participant percentiles
      - proxy_signal: error_proxy_binary from error proxy script
                      (objective behavioral, immune to subjective rating noise)

    Voting:
      - n_high = count of signals == 1
      - n_low  = count of signals == 0
      - label  = 1  if n_high >= 2 and n_low == 0
                 0  if n_low  >= 2 and n_high == 0
                 nan otherwise (true ambiguity → drop)

    Confidence weight for the row:
      - 3/3 agree (no nan, no disagree)        → weight 1.00
      - 2/3 agree, 1 abstain (nan signal)     → weight 0.70
      - 2/3 agree, 1 disagrees                → weight 0.50
      - else                                   → label is nan, weight irrelevant

    Returns (df_with_labels, weights) where weights aligns with df rows.
    """
    df = df.copy()

    # Build the proxy error binary using the existing utility (uses raw error
    # cols not smoothed — error counts have less noise than ratings).
    df = create_proxy_error_binary_target_per_participant(
        df=df,
        group_col=group_col,
        proxy_cols=PROXY_ERROR_COLS,
        lower_percentile=low_pctl,
        upper_percentile=high_pctl,
        new_col=PROXY_TARGET_COL,
    )

    tlx_sig = _per_pid_binary_signal(df, group_col, TLX_SMOOTH_COL, low_pctl, high_pctl)
    sleep_sig = _per_pid_binary_signal(df, group_col, SLEEP_SMOOTH_COL, low_pctl, high_pctl)
    proxy_sig = df[PROXY_TARGET_COL].to_numpy(dtype=float)

    sigs = np.stack([tlx_sig, sleep_sig, proxy_sig], axis=1)  # (n, 3)
    n_high = np.nansum(sigs == 1.0, axis=1)
    n_low = np.nansum(sigs == 0.0, axis=1)
    n_present = np.sum(~np.isnan(sigs), axis=1)

    label = np.full(len(df), np.nan)
    weight = np.zeros(len(df), dtype=float)

    for i in range(len(df)):
        h, lo_, present = int(n_high[i]), int(n_low[i]), int(n_present[i])
        if h >= 2 and lo_ == 0:
            label[i] = 1.0
            if h == 3:
                weight[i] = 1.00
            elif present == 3 and lo_ == 1:  # 2 high, 1 low
                weight[i] = 0.50
            else:  # 2 high, 1 abstain
                weight[i] = 0.70
        elif lo_ >= 2 and h == 0:
            label[i] = 0.0
            if lo_ == 3:
                weight[i] = 1.00
            elif present == 3 and h == 1:  # 2 low, 1 high
                weight[i] = 0.50
            else:
                weight[i] = 0.70
        # else: label stays nan

    df[BINARY_TARGET_COL] = label
    df[WEIGHT_COL] = weight
    # Keep modality signals on the dataframe for validation comparisons
    df[TLX_SIGNAL_COL] = tlx_sig
    df[SLEEP_SIGNAL_COL] = sleep_sig
    df[PROXY_SIGNAL_COL] = proxy_sig

    # Audit table: row counts in each agreement bucket
    buckets = []
    for h_, lo_ in [(3, 0), (2, 0), (2, 1), (0, 3), (0, 2), (1, 2)]:
        n = int(np.sum((n_high == h_) & (n_low == lo_)))
        if n > 0:
            buckets.append({"n_high": h_, "n_low": lo_, "rows": n})
    audit = pd.DataFrame(buckets)
    return df, weight, audit


# ─────────────────────────────────────────────────────────────────────────────
# 3. Soft regression target (continuous denoised score per participant)
# ─────────────────────────────────────────────────────────────────────────────

def attach_soft_target(df: pd.DataFrame, group_col: str = GROUP_COL) -> pd.DataFrame:
    """Per-participant min-max scale of the smoothed alertness score → [0, 1].
    This is the regression target for Head B."""
    df = df.copy()
    soft = np.full(len(df), np.nan)
    for _, idx in df.groupby(group_col).groups.items():
        idx = np.asarray(list(idx))
        v = df.loc[idx, SCORE_SMOOTH_COL].to_numpy(dtype=float)
        valid = ~np.isnan(v)
        if valid.sum() < 2:
            continue
        lo, hi = float(np.min(v[valid])), float(np.max(v[valid]))
        if hi - lo < 1e-9:
            soft[idx] = 0.5
        else:
            soft[idx[valid]] = (v[valid] - lo) / (hi - lo)
    df[SOFT_TARGET_COL] = soft
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Preprocessor + LGBM helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor(numeric_features, categorical_features, scale_numeric=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    num_t = Pipeline(num_steps)
    cat_t = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_t, numeric_features),
        ("cat", cat_t, categorical_features),
    ])


def build_lgbm_classifier(seed: int = RANDOM_STATE):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1, **LGBM_PARAMS_CLF)


def build_lgbm_regressor(seed: int = RANDOM_STATE):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1, **LGBM_PARAMS_REG)


def build_clf_pipeline(numeric_features, categorical_features, seed: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, categorical_features)),
        ("model", build_lgbm_classifier(seed)),
    ])


def build_reg_pipeline(numeric_features, categorical_features, seed: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, categorical_features)),
        ("model", build_lgbm_regressor(seed)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP feature selection (regressor on soft labels — uses ALL rows)
# ─────────────────────────────────────────────────────────────────────────────

def _get_transformed_feature_names(preprocessor, numeric_features, categorical_features):
    names = list(numeric_features)
    if categorical_features:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = (
            onehot.get_feature_names_out(categorical_features)
            if hasattr(onehot, "get_feature_names_out")
            else onehot.get_feature_names(categorical_features)
        )
        names.extend(list(cat_names))
    return names


def _map_to_raw(name, numeric_features, categorical_features):
    if name in numeric_features:
        return name
    for col in categorical_features:
        if name.startswith(f"{col}_"):
            return col
    return name


def shap_select_on_soft(
    df: pd.DataFrame,
    feature_cols: List[str],
    soft_target_col: str,
    group_col: str,
    keep_ratio: float,
    min_features: int,
    sample_size: int = 1500,
) -> Tuple[List[str], pd.DataFrame]:
    """Feature selection using LGBM regressor on the SOFT target.
    Soft target is defined for nearly all rows → richer SHAP signal than
    selecting on the agreement-filtered hard label (which drops ~half the rows).
    """
    cols = list(dict.fromkeys(feature_cols + [soft_target_col, group_col]))
    sub = df[cols].dropna(subset=[soft_target_col, group_col]).reset_index(drop=True)
    X = sub[feature_cols]
    y = sub[soft_target_col].to_numpy()
    groups = sub[group_col]

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    pre = build_preprocessor(numeric_features, categorical_features)
    base = build_lgbm_regressor()
    pipe = Pipeline([("preprocess", pre), ("model", base)])
    pipe.fit(X_scaled, y)

    rng = np.random.RandomState(RANDOM_STATE)
    if len(X_scaled) > sample_size:
        idx = rng.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled.iloc[idx]
    else:
        X_sample = X_scaled
    X_trans = pre.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    transformed_names = _get_transformed_feature_names(pre, numeric_features, categorical_features)
    df_imp = pd.DataFrame({"transformed_feature": transformed_names, "mean_abs_shap": mean_abs})
    df_imp["raw_feature"] = df_imp["transformed_feature"].apply(
        lambda n: _map_to_raw(n, numeric_features, categorical_features)
    )
    raw = (
        df_imp.groupby("raw_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    n_total = len(raw)
    n_keep = max(min_features, ceil(n_total * keep_ratio))
    n_keep = min(n_keep, n_total)
    return raw["raw_feature"].head(n_keep).tolist(), raw


# ─────────────────────────────────────────────────────────────────────────────
# 6. Two-head stacked CV — Head A clf + Head B regressor → LogReg meta
# ─────────────────────────────────────────────────────────────────────────────

def _confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return tp, fp, fn, tn, p, r, f1


def _train_oof_heads(
    X_tr: pd.DataFrame,
    y_tr_hard: np.ndarray,
    y_tr_soft: np.ndarray,
    sw_tr: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Produce OOF predictions on training set for Head A (clf) and Head B (reg).
    Used as level-2 inputs without leakage."""
    oof_a = np.full(len(y_tr_hard), 0.5, dtype=float)
    oof_b = np.full(len(y_tr_soft), 0.5, dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    # Stratify on hard label for inner CV
    for tr, va in cv.split(X_tr, y_tr_hard):
        if len(np.unique(y_tr_hard[tr])) < 2:
            continue
        # Head A: classifier on hard label, weighted
        pa = build_clf_pipeline(numeric_features, categorical_features)
        pa.fit(X_tr.iloc[tr], y_tr_hard[tr], model__sample_weight=sw_tr[tr])
        oof_a[va] = pa.predict_proba(X_tr.iloc[va])[:, 1]
        # Head B: regressor on soft label, all rows in tr
        pb = build_reg_pipeline(numeric_features, categorical_features)
        pb.fit(X_tr.iloc[tr], y_tr_soft[tr])
        oof_b[va] = np.clip(pb.predict(X_tr.iloc[va]), 0.0, 1.0)
    return oof_a, oof_b


def cross_cv_two_head_stacked(
    df: pd.DataFrame,
    feature_cols: List[str],
    hard_target_col: str,
    soft_target_col: str,
    weight_col: str,
    group_col: str,
    meta_extra_features: List[str],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Cross-participant CV (LOPO). Train two heads on TRAIN with OOF inner CV
    to produce level-2 inputs, then a LogReg meta-learner on top."""
    cols = list(dict.fromkeys(
        feature_cols + [hard_target_col, soft_target_col, weight_col, group_col]
    ))
    sub = df[cols].copy()
    # Keep rows with EITHER a hard label OR a soft target (both heads can train on them)
    sub = sub[sub[soft_target_col].notna() & sub[group_col].notna()].reset_index(drop=True)

    X = sub[feature_cols]
    y_hard = sub[hard_target_col].to_numpy(dtype=float)  # may have NaN
    y_soft = sub[soft_target_col].to_numpy(dtype=float)
    weight = sub[weight_col].to_numpy(dtype=float)
    groups = sub[group_col]

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    logo = LeaveOneGroupOut()
    n_groups = len(np.unique(groups))
    print(f"    Cross-CV (two-head stacked LOPO over {n_groups} participants)")

    total_tp = total_fp = total_fn = total_tn = 0
    all_yt: List[np.ndarray] = []
    all_ys: List[np.ndarray] = []

    meta_extra_present = [f for f in meta_extra_features
                          if f in X_scaled.columns and pd.api.types.is_numeric_dtype(X_scaled[f])]

    for fold_i, (tr, te) in enumerate(logo.split(X_scaled, y_soft, groups), start=1):
        # Test rows must have hard label to be evaluated; if none, skip
        te_has_label = ~np.isnan(y_hard[te])
        if te_has_label.sum() == 0:
            if fold_i % 5 == 0 or fold_i == n_groups:
                print(f"      fold {fold_i}/{n_groups} skipped (no labelled test rows)")
            continue
        te_eval = te[te_has_label]

        # Train rows: only those with hard label, for Head A weighting/stratify;
        # but Head B uses all rows with soft label (more data).
        tr_hard_mask = ~np.isnan(y_hard[tr])
        if tr_hard_mask.sum() < 10 or len(np.unique(y_hard[tr][tr_hard_mask])) < 2:
            continue
        tr_hard = tr[tr_hard_mask]

        X_tr_hard = X_scaled.iloc[tr_hard]
        y_tr_hard = y_hard[tr_hard].astype(int)
        y_tr_soft_for_clf_idx = y_soft[tr_hard]
        sw_tr = weight[tr_hard]
        sw_tr = np.where(sw_tr > 0, sw_tr, 1.0)  # safety

        # OOF level-1 outputs on tr_hard (so meta can be trained without leakage)
        oof_a, oof_b = _train_oof_heads(
            X_tr_hard, y_tr_hard, y_tr_soft_for_clf_idx, sw_tr,
            numeric_features, categorical_features, n_splits=5,
        )

        # Full level-1 fits for test prediction
        full_a = build_clf_pipeline(numeric_features, categorical_features)
        full_a.fit(X_tr_hard, y_tr_hard, model__sample_weight=sw_tr)
        proba_a_te = full_a.predict_proba(X_scaled.iloc[te_eval])[:, 1]

        # Head B trains on ALL training rows that have a soft target (more data)
        full_b = build_reg_pipeline(numeric_features, categorical_features)
        full_b.fit(X_scaled.iloc[tr], y_soft[tr])
        score_b_te = np.clip(full_b.predict(X_scaled.iloc[te_eval]), 0.0, 1.0)

        # Level 2: LogReg meta on [oof_a, oof_b, top SHAP raw features (numeric)]
        if meta_extra_present:
            meta_imp = SimpleImputer(strategy="median")
            meta_scl = StandardScaler()
            X_meta_tr_extra_raw = X_tr_hard[meta_extra_present].to_numpy()
            X_meta_te_extra_raw = X_scaled.iloc[te_eval][meta_extra_present].to_numpy()
            X_meta_tr_extra = meta_scl.fit_transform(meta_imp.fit_transform(X_meta_tr_extra_raw))
            X_meta_te_extra = meta_scl.transform(meta_imp.transform(X_meta_te_extra_raw))
            X_meta_tr = np.column_stack([oof_a.reshape(-1, 1), oof_b.reshape(-1, 1), X_meta_tr_extra])
            X_meta_te = np.column_stack([proba_a_te.reshape(-1, 1), score_b_te.reshape(-1, 1), X_meta_te_extra])
        else:
            X_meta_tr = np.column_stack([oof_a.reshape(-1, 1), oof_b.reshape(-1, 1)])
            X_meta_te = np.column_stack([proba_a_te.reshape(-1, 1), score_b_te.reshape(-1, 1)])

        meta = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
        meta.fit(X_meta_tr, y_tr_hard, sample_weight=sw_tr)
        proba_te = meta.predict_proba(X_meta_te)[:, 1]

        y_te_eval = y_hard[te_eval].astype(int)
        y_pred = (proba_te >= 0.5).astype(int)
        tp, fp, fn, tn, _, _, _ = _confusion(y_te_eval, y_pred)
        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
        all_yt.append(y_te_eval)
        all_ys.append(proba_te)

        if fold_i % 5 == 0 or fold_i == n_groups:
            print(f"      fold {fold_i}/{n_groups} done")

    if total_tp + total_fp + total_fn + total_tn == 0:
        return {}, np.array([], int), np.array([], float)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return (
        {"TP": float(total_tp), "FP": float(total_fp), "FN": float(total_fn), "TN": float(total_tn),
         "precision": float(p), "recall": float(r), "f1": float(f1)},
        np.concatenate(all_yt),
        np.concatenate(all_ys),
    )


def within_cv_two_head(
    df: pd.DataFrame,
    feature_cols: List[str],
    hard_target_col: str,
    soft_target_col: str,
    weight_col: str,
    group_col: str,
    n_splits_default: int = 5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Within-participant CV. Per-PID folds are tiny → skip stacking, just
    average the two heads' probabilities (Head A clf + Head B regressor)."""
    cols = list(dict.fromkeys(
        feature_cols + [hard_target_col, soft_target_col, weight_col, group_col]
    ))
    sub = df[cols].copy()
    sub = sub[sub[hard_target_col].notna() & sub[group_col].notna()].reset_index(drop=True)
    X = sub[feature_cols]
    y_hard = sub[hard_target_col].to_numpy().astype(int)
    y_soft = sub[soft_target_col].to_numpy(dtype=float)
    weight = sub[weight_col].to_numpy(dtype=float)
    groups = sub[group_col]

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    total_tp = total_fp = total_fn = total_tn = 0
    all_yt: List[np.ndarray] = []
    all_ys: List[np.ndarray] = []
    all_groups: List[np.ndarray] = []

    for pid, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx))
        if idx.size < 3:
            continue
        X_p = X_scaled.iloc[idx]
        y_p_hard = y_hard[idx]
        y_p_soft = y_soft[idx]
        w_p = weight[idx]
        cc = np.bincount(y_p_hard, minlength=2)
        min_class = int(cc[cc > 0].min()) if np.any(cc > 0) else 0
        n_splits = min(n_splits_default, idx.size, min_class)
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for tr, te in cv.split(X_p, y_p_hard):
            if len(np.unique(y_p_hard[tr])) < 2:
                continue
            sw = np.where(w_p[tr] > 0, w_p[tr], 1.0)

            pa = build_clf_pipeline(numeric_features, categorical_features)
            pa.fit(X_p.iloc[tr], y_p_hard[tr], model__sample_weight=sw)
            proba_a = pa.predict_proba(X_p.iloc[te])[:, 1]

            pb = build_reg_pipeline(numeric_features, categorical_features)
            # Soft target may have NaN; impute to 0.5 for safety
            ys_tr = np.where(np.isnan(y_p_soft[tr]), 0.5, y_p_soft[tr])
            pb.fit(X_p.iloc[tr], ys_tr)
            score_b = np.clip(pb.predict(X_p.iloc[te]), 0.0, 1.0)

            proba = 0.5 * proba_a + 0.5 * score_b
            y_pred = (proba >= 0.5).astype(int)
            tp, fp, fn, tn, _, _, _ = _confusion(y_p_hard[te], y_pred)
            total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
            all_yt.append(y_p_hard[te])
            all_ys.append(proba)
            all_groups.append(np.full(y_p_hard[te].shape, pid))

    if total_tp + total_fp + total_fn + total_tn == 0:
        return {}, np.array([], int), np.array([], float), np.array([])
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return (
        {"TP": float(total_tp), "FP": float(total_fp), "FN": float(total_fn), "TN": float(total_tn),
         "precision": float(p), "recall": float(r), "f1": float(f1)},
        np.concatenate(all_yt),
        np.concatenate(all_ys),
        np.concatenate(all_groups),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Threshold optimization + reporting
# ─────────────────────────────────────────────────────────────────────────────

def best_threshold_global(y_true, y_score, precision_floor=0.0):
    if y_true.size == 0 or y_score.size == 0:
        return {"best_threshold": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan")}
    best = None; fb = None
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_score >= t).astype(int)
        _, _, _, _, p, r, f = _confusion(y_true, y_pred)
        row = {"best_threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)}
        if fb is None or row["f1"] > fb["f1"]:
            fb = row
        if p >= precision_floor and (best is None or row["f1"] > best["f1"]):
            best = row
    return best if best is not None else fb


def per_pid_threshold_metrics(y_true, y_score, groups):
    """In-sample per-PID best threshold. INFLATED — for upper bound only."""
    total_tp = total_fp = total_fn = total_tn = 0
    for pid in np.unique(groups):
        m = groups == pid
        y_t = y_true[m]; y_s = y_score[m]
        if y_t.size < 4 or len(np.unique(y_t)) < 2:
            y_pred = (y_s >= 0.5).astype(int)
        else:
            best_f1 = -1.0; best_pred = (y_s >= 0.5).astype(int)
            for t in np.linspace(0.1, 0.9, 81):
                pred = (y_s >= t).astype(int)
                _, _, _, _, _, _, f = _confusion(y_t, pred)
                if f > best_f1:
                    best_f1 = f; best_pred = pred
            y_pred = best_pred
        tp, fp, fn, tn, _, _, _ = _confusion(y_t, y_pred)
        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"TP": float(total_tp), "FP": float(total_fp), "FN": float(total_fn), "TN": float(total_tn),
            "precision": float(p), "recall": float(r), "f1": float(f1)}


def _compute_pr_auc(y_true, y_score):
    if y_true.size == 0 or y_score.size == 0:
        return float("nan")
    p, r, _ = precision_recall_curve(y_true, y_score)
    return float(auc(r, p))


def _print_block(title, m):
    print(f"  {title}:")
    if not m:
        print("    (no valid folds)"); return
    print(f"    TP={m['TP']:.0f} FP={m['FP']:.0f} FN={m['FN']:.0f} TN={m['TN']:.0f}")
    print(f"    precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Validation experiments — make the headline result bullet-proof
#
# Three independent validations to test whether the F1=0.905 cross result is
# a real generalisation gain or an artifact of selection bias:
#
#   V1. Predict on the dropped 528 rows (no agreement label) and compare
#       predictions to alternative reference labels (old combined TLX+sleepiness
#       cut, individual modality signals). If the model still classifies
#       correctly off the easy set → genuine generalisation.
#
#   V2. Re-run LOPO with SHAP feature selection inside each fold (no test data
#       sees the SHAP estimator). Removes the largest legitimate leakage path.
#
#   V3. Train and evaluate on the SAME 375 rows but using the OLD label scheme
#       (combined TLX+sleepiness, same percentile cuts). Isolates the gain
#       from the labeling protocol vs the row selection.
# ─────────────────────────────────────────────────────────────────────────────

def _make_old_combined_label(
    df: pd.DataFrame,
    group_col: str,
    low_pctl: float,
    high_pctl: float,
    out_col: str,
) -> pd.DataFrame:
    """Old labeling scheme: per-participant percentile cut on raw TLX+sleepiness
    (no smoothing, no proxy errors). Same logic as the original
    `create_alertness_binary_target_per_participant` but kept inline so we can
    apply it independently of the agreement label."""
    df = df.copy()
    score = df[TLX_COL] + df[SLEEPINESS_COL]
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby(group_col).groups.items():
        idx = np.asarray(list(idx))
        v = score.iloc[idx].to_numpy(dtype=float)
        valid = ~np.isnan(v)
        if valid.sum() == 0:
            continue
        lo = np.quantile(v[valid], low_pctl)
        hi = np.quantile(v[valid], high_pctl)
        for i, x in zip(idx, v):
            if np.isnan(x):
                continue
            if x <= lo:
                out[i] = 0.0
            elif x >= hi:
                out[i] = 1.0
    df[out_col] = out
    return df


def _evaluate_predictions_against_label(
    y_score: np.ndarray,
    y_ref: np.ndarray,
    label_name: str,
) -> Dict[str, float]:
    """Evaluate model predictions against a reference binary label.
    Returns a row of metrics; rows where reference is NaN are skipped."""
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    valid = ~np.isnan(y_ref)
    n_valid = int(valid.sum())
    if n_valid == 0 or len(np.unique(y_ref[valid])) < 2:
        return {"label": label_name, "n": n_valid, "f1@0.50": float("nan"),
                "precision@0.50": float("nan"), "recall@0.50": float("nan"),
                "accuracy@0.50": float("nan"), "kappa@0.50": float("nan"),
                "pr_auc": float("nan"), "roc_auc": float("nan"),
                "f1_best_t": float("nan"), "best_threshold": float("nan")}
    yt = y_ref[valid].astype(int)
    ys = y_score[valid]
    yp = (ys >= 0.5).astype(int)
    _, _, _, _, p, r, f1 = _confusion(yt, yp)
    accuracy = float((yp == yt).mean())
    kappa = float(cohen_kappa_score(yt, yp))
    pr_auc = _compute_pr_auc(yt, ys)
    try:
        roc = float(roc_auc_score(yt, ys))
    except Exception:
        roc = float("nan")
    best = best_threshold_global(yt, ys, precision_floor=0.0)
    return {
        "label": label_name,
        "n": n_valid,
        "n_pos": int(np.sum(yt == 1)),
        "n_neg": int(np.sum(yt == 0)),
        "f1@0.50": float(f1),
        "precision@0.50": float(p),
        "recall@0.50": float(r),
        "accuracy@0.50": accuracy,
        "kappa@0.50": kappa,
        "pr_auc": pr_auc,
        "roc_auc": roc,
        "f1_best_t": float(best["f1"]),
        "best_threshold": float(best["best_threshold"]),
    }


# ─── V1: Extrapolate to dropped rows ─────────────────────────────────────────

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
    """V1. Train two-head stacked model on agreement-labeled rows; predict on
    dropped rows; compare predictions against alternative reference labels.

    Returns a comparison table (one row per reference label).
    Predictions are also evaluated on the labeled set as a sanity check
    (training-set fit; only useful as a calibration reference, not for
    generalisation claims).
    """
    print("\n[V1] Extrapolation to dropped rows")
    # Build OLD combined label for ALL rows (not affected by agreement filter)
    df_v1 = _make_old_combined_label(df, group_col, low_pctl, high_pctl, OLD_TARGET_COL)

    cols = list(dict.fromkeys(
        selected_features + [hard_target_col, soft_target_col, weight_col, group_col,
                             OLD_TARGET_COL, TLX_SIGNAL_COL, SLEEP_SIGNAL_COL, PROXY_SIGNAL_COL]
    ))
    sub = df_v1[cols].copy().reset_index(drop=True)

    # Need the full feature subset to fit per-PID IQR scaler on all rows that
    # have feature data; subset by group_col only.
    sub = sub[sub[group_col].notna()].reset_index(drop=True)

    X = sub[selected_features]
    groups = sub[group_col]
    numeric_features = [c for c in selected_features if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in selected_features if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    y_hard = sub[hard_target_col].to_numpy(dtype=float)
    y_soft = sub[soft_target_col].to_numpy(dtype=float)
    weight = sub[weight_col].to_numpy(dtype=float)

    labeled_mask = ~np.isnan(y_hard)
    dropped_mask = ~labeled_mask
    n_labeled = int(labeled_mask.sum())
    n_dropped = int(dropped_mask.sum())
    print(f"  Train on {n_labeled} labelled rows; predict on {n_dropped} dropped rows")

    # Fit Head A on labeled rows (weighted)
    sw_lab = np.where(weight[labeled_mask] > 0, weight[labeled_mask], 1.0)
    full_a = build_clf_pipeline(numeric_features, categorical_features)
    full_a.fit(X_scaled[labeled_mask], y_hard[labeled_mask].astype(int), model__sample_weight=sw_lab)

    # Fit Head B on rows with soft target available (most rows)
    soft_mask = ~np.isnan(y_soft)
    full_b = build_reg_pipeline(numeric_features, categorical_features)
    full_b.fit(X_scaled[soft_mask], y_soft[soft_mask])

    # Predict on dropped rows
    proba_a_drop = full_a.predict_proba(X_scaled[dropped_mask])[:, 1]
    score_b_drop = np.clip(full_b.predict(X_scaled[dropped_mask]), 0.0, 1.0)
    # Simple 50/50 average — the level-2 LogReg meta requires LOPO OOF features
    # which we don't have here for a single-fit final model. The average is a
    # reasonable proxy.
    proba_drop = 0.5 * proba_a_drop + 0.5 * score_b_drop

    # Reference labels on dropped rows
    refs = {
        "old_combined_TLX_plus_sleepiness (33/67)": sub.loc[dropped_mask, OLD_TARGET_COL].to_numpy(),
        "tlx_signal_only (smoothed, 33/67)": sub.loc[dropped_mask, TLX_SIGNAL_COL].to_numpy(),
        "sleepiness_signal_only (smoothed, 33/67)": sub.loc[dropped_mask, SLEEP_SIGNAL_COL].to_numpy(),
        "proxy_errors_signal_only (33/67)": sub.loc[dropped_mask, PROXY_SIGNAL_COL].to_numpy(),
    }
    rows = []
    for name, ref in refs.items():
        rows.append(_evaluate_predictions_against_label(proba_drop, ref, name))

    # Also report on FULL data (sanity check; trained on these labels — optimistic)
    proba_a_all = full_a.predict_proba(X_scaled)[:, 1]
    score_b_all = np.clip(full_b.predict(X_scaled), 0.0, 1.0)
    proba_all = 0.5 * proba_a_all + 0.5 * score_b_all
    full_refs = {
        "ALL rows vs old_combined (sanity, train+test mixed)": sub[OLD_TARGET_COL].to_numpy(),
        "ALL rows vs agreement label (sanity, includes train)": sub[hard_target_col].to_numpy(),
    }
    for name, ref in full_refs.items():
        rows.append(_evaluate_predictions_against_label(proba_all, ref, name))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


# ─── V2: Cross-CV with SHAP selection inside each fold ───────────────────────

def cross_cv_two_head_stacked_inner_shap(
    df: pd.DataFrame,
    extended_features: List[str],
    hard_target_col: str,
    soft_target_col: str,
    weight_col: str,
    group_col: str,
    keep_ratio: float,
    min_features: int,
    meta_top_k: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, pd.DataFrame]:
    """V2. LOPO where SHAP feature selection runs on the TRAINING fold only.
    No test participant's rows ever see the SHAP estimator. This is the
    leakage-free version of the cross-CV result."""
    print("\n[V2] Cross-CV with SHAP feature selection inside each fold")
    cols = list(dict.fromkeys(
        extended_features + [hard_target_col, soft_target_col, weight_col, group_col]
    ))
    sub = df[cols].copy()
    sub = sub[sub[soft_target_col].notna() & sub[group_col].notna()].reset_index(drop=True)

    y_hard = sub[hard_target_col].to_numpy(dtype=float)
    y_soft = sub[soft_target_col].to_numpy(dtype=float)
    weight = sub[weight_col].to_numpy(dtype=float)
    groups = sub[group_col]

    logo = LeaveOneGroupOut()
    n_groups = len(np.unique(groups))
    print(f"    Inner-SHAP LOPO over {n_groups} participants (slow — SHAP per fold)")

    total_tp = total_fp = total_fn = total_tn = 0
    all_yt: List[np.ndarray] = []
    all_ys: List[np.ndarray] = []
    fold_records: List[Dict] = []

    # Build group iterator with stable order
    fold_iter = list(logo.split(sub, y_soft, groups))

    for fold_i, (tr, te) in enumerate(fold_iter, start=1):
        te_has_label = ~np.isnan(y_hard[te])
        if te_has_label.sum() == 0:
            continue
        te_eval = te[te_has_label]

        tr_hard_mask = ~np.isnan(y_hard[tr])
        if tr_hard_mask.sum() < 10 or len(np.unique(y_hard[tr][tr_hard_mask])) < 2:
            continue
        tr_hard = tr[tr_hard_mask]

        # SHAP selection on TRAINING rows only
        df_tr_for_shap = sub.iloc[tr].copy()
        try:
            selected_in_fold, _ = shap_select_on_soft(
                df_tr_for_shap, extended_features,
                soft_target_col=soft_target_col, group_col=group_col,
                keep_ratio=keep_ratio, min_features=min_features,
            )
        except Exception as exc:
            print(f"      fold {fold_i}/{n_groups} SHAP failed: {exc}")
            continue

        # Per-fold preprocessing on selected features
        X_fold = sub[selected_in_fold]
        numeric_features = [c for c in selected_in_fold if pd.api.types.is_numeric_dtype(X_fold[c])]
        categorical_features = [c for c in selected_in_fold if c not in numeric_features]
        scaler_stats = fit_participant_iqr_scaler(X_fold, groups, numeric_features)
        X_scaled = transform_with_participant_iqr_scaler(X_fold, groups, numeric_features, *scaler_stats)

        X_tr_hard = X_scaled.iloc[tr_hard]
        y_tr_hard = y_hard[tr_hard].astype(int)
        y_tr_soft_for_clf = y_soft[tr_hard]
        sw_tr = np.where(weight[tr_hard] > 0, weight[tr_hard], 1.0)

        oof_a, oof_b = _train_oof_heads(
            X_tr_hard, y_tr_hard, y_tr_soft_for_clf, sw_tr,
            numeric_features, categorical_features, n_splits=5,
        )

        full_a = build_clf_pipeline(numeric_features, categorical_features)
        full_a.fit(X_tr_hard, y_tr_hard, model__sample_weight=sw_tr)
        proba_a_te = full_a.predict_proba(X_scaled.iloc[te_eval])[:, 1]

        full_b = build_reg_pipeline(numeric_features, categorical_features)
        full_b.fit(X_scaled.iloc[tr], y_soft[tr])
        score_b_te = np.clip(full_b.predict(X_scaled.iloc[te_eval]), 0.0, 1.0)

        meta_extra_present = [f for f in selected_in_fold[:meta_top_k]
                              if f in X_scaled.columns and pd.api.types.is_numeric_dtype(X_scaled[f])]
        if meta_extra_present:
            meta_imp = SimpleImputer(strategy="median")
            meta_scl = StandardScaler()
            X_meta_tr_extra = meta_scl.fit_transform(
                meta_imp.fit_transform(X_tr_hard[meta_extra_present].to_numpy())
            )
            X_meta_te_extra = meta_scl.transform(
                meta_imp.transform(X_scaled.iloc[te_eval][meta_extra_present].to_numpy())
            )
            X_meta_tr = np.column_stack([oof_a.reshape(-1, 1), oof_b.reshape(-1, 1), X_meta_tr_extra])
            X_meta_te = np.column_stack([proba_a_te.reshape(-1, 1), score_b_te.reshape(-1, 1), X_meta_te_extra])
        else:
            X_meta_tr = np.column_stack([oof_a.reshape(-1, 1), oof_b.reshape(-1, 1)])
            X_meta_te = np.column_stack([proba_a_te.reshape(-1, 1), score_b_te.reshape(-1, 1)])

        meta = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
        meta.fit(X_meta_tr, y_tr_hard, sample_weight=sw_tr)
        proba_te = meta.predict_proba(X_meta_te)[:, 1]

        y_te_eval = y_hard[te_eval].astype(int)
        y_pred = (proba_te >= 0.5).astype(int)
        tp, fp, fn, tn, _, _, _ = _confusion(y_te_eval, y_pred)
        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
        all_yt.append(y_te_eval)
        all_ys.append(proba_te)
        fold_records.append({
            "fold": fold_i,
            "n_test": int(len(te_eval)),
            "n_features_selected": len(selected_in_fold),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

        if fold_i % 5 == 0 or fold_i == n_groups:
            print(f"      fold {fold_i}/{n_groups} done  "
                  f"(selected {len(selected_in_fold)} feats)")

    if total_tp + total_fp + total_fn + total_tn == 0:
        return {}, np.array([], int), np.array([], float), pd.DataFrame()
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    metrics = {
        "TP": float(total_tp), "FP": float(total_fp), "FN": float(total_fn), "TN": float(total_tn),
        "precision": float(p), "recall": float(r), "f1": float(f1),
    }
    return (
        metrics,
        np.concatenate(all_yt),
        np.concatenate(all_ys),
        pd.DataFrame(fold_records),
    )


# ─── V3: Same rows, OLD label scheme ─────────────────────────────────────────

def validation_old_label_on_same_rows(
    df: pd.DataFrame,
    selected_features: List[str],
    hard_target_col: str,
    soft_target_col: str,
    weight_col: str,
    group_col: str,
    low_pctl: float,
    high_pctl: float,
    precision_floor: float,
    meta_top_k: int,
) -> Dict[str, Dict[str, float]]:
    """V3. Use the OLD label scheme (combined TLX+sleepiness percentile cut)
    on the SAME rows that received an agreement label. Isolates whether the
    headline gain comes from row selection (easy rows) or from the labeling
    protocol itself (multi-modal agreement).

    Returns dict with within and cross metrics for the OLD-label experiment.
    """
    print("\n[V3] Same rows, OLD label scheme (combined TLX+sleepiness cut)")
    df_v3 = _make_old_combined_label(df, group_col, low_pctl, high_pctl, OLD_TARGET_COL)

    # Filter to rows that BOTH (a) received an agreement label, and
    # (b) have an OLD label (some agreement-labeled rows may sit in the OLD
    # middle band because TLX+sleepiness alone didn't pick them up).
    mask = df_v3[hard_target_col].notna() & df_v3[OLD_TARGET_COL].notna()
    n_intersection = int(mask.sum())
    n_agreement = int(df_v3[hard_target_col].notna().sum())
    print(f"  Intersection (both labels available): {n_intersection} / {n_agreement} agreement rows")
    if n_intersection < 50:
        print("  (too few rows for meaningful comparison)")
        return {}

    # Replace BINARY_TARGET_COL with OLD label, set all weights to 1.0
    df_v3_subset = df_v3.loc[mask].copy().reset_index(drop=True)
    df_v3_subset[hard_target_col] = df_v3_subset[OLD_TARGET_COL]
    df_v3_subset[weight_col] = 1.0
    bal = pd.Series(df_v3_subset[hard_target_col]).value_counts(normalize=True).sort_index()
    print(f"  OLD-label balance on intersection: {bal.to_dict()}")
    # Agreement vs OLD: how often do they disagree on the same rows?
    agreement_label = df_v3.loc[mask, hard_target_col].to_numpy(dtype=float)
    # Recover the original agreement label which was overwritten above; pull from df_v3
    # (We saved the OLD label in OLD_TARGET_COL, agreement label in hard_target_col before overwrite)
    # df_v3.loc[mask, hard_target_col] is still the agreement label since we modified df_v3_subset, not df_v3
    n_disagree = int(np.sum(df_v3.loc[mask, hard_target_col].to_numpy() != df_v3.loc[mask, OLD_TARGET_COL].to_numpy()))
    print(f"  Agreement vs OLD disagree on {n_disagree} / {n_intersection} rows "
          f"({n_disagree / n_intersection:.1%})")

    # Within CV with OLD label
    print("  Within-CV (OLD label, same rows)...")
    wp_m, wp_yt, wp_ys, _ = within_cv_two_head(
        df_v3_subset, selected_features, hard_target_col, soft_target_col,
        weight_col, group_col,
    )
    wp_pr = _compute_pr_auc(wp_yt, wp_ys)
    wp_g = best_threshold_global(wp_yt, wp_ys, precision_floor=precision_floor)
    _print_block("  Within @ 0.50", wp_m)
    print(f"      PR-AUC={wp_pr:.3f}  global-t f1={wp_g['f1']:.3f} (t={wp_g['best_threshold']:.2f})")

    # Cross CV with OLD label
    print("  Cross-CV (OLD label, same rows)...")
    cp_m, cp_yt, cp_ys = cross_cv_two_head_stacked(
        df_v3_subset, selected_features, hard_target_col, soft_target_col,
        weight_col, group_col, meta_extra_features=selected_features[:meta_top_k],
    )
    cp_pr = _compute_pr_auc(cp_yt, cp_ys)
    cp_g = best_threshold_global(cp_yt, cp_ys, precision_floor=precision_floor)
    _print_block("  Cross @ 0.50", cp_m)
    print(f"      PR-AUC={cp_pr:.3f}  global-t f1={cp_g['f1']:.3f} (t={cp_g['best_threshold']:.2f})")

    return {
        "within": {**wp_m, "pr_auc": wp_pr,
                   "f1_best_t": wp_g["f1"], "best_threshold": wp_g["best_threshold"]},
        "cross": {**cp_m, "pr_auc": cp_pr,
                  "f1_best_t": cp_g["f1"], "best_threshold": cp_g["best_threshold"]},
        "n_intersection": n_intersection,
        "n_disagree_with_agreement": n_disagree,
    }


# ─── V4: LOPO predictions on ALL rows (agreed + disagreed) ───────────────────

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
    """V4. Generate LOPO predictions for ALL rows (both agreement-labeled
    and dropped/disagreed) — each row predicted by a model that has never
    seen that participant — and evaluate against several reference labels.

    Critical difference vs V1: V1 fits ONE model on all labeled rows and
    predicts on dropped rows (in-sample for labeled, out-of-sample for
    dropped). V4 cross-validates LOPO so every prediction is out-of-sample
    for its participant.

    REFERENCE LABEL DESIGN — addresses cross-cut comparability:

      Cut-MATCHED references (built with low_pctl/high_pctl from the
      agreement-label cut): rows in the middle band have NaN reference and
      are EXCLUDED from F1 calculation. Evaluable row set differs between
      33/67 and 49/51 runs, so F1 numbers are NOT directly comparable.

      Cut-UNIVERSAL references (built at fixed 50/50 median split per
      participant): every row gets a 0 or 1 reference label. SAME evaluable
      row set across all runs → numbers ARE directly comparable.

      The 50/50 universal numbers are the cleanest single answer to
      "how well does the model do on the full dataset, independent of
      which percentile we used to build the agreement labels?"

    Returns (eval_table, prediction_table).
    """
    print("\n[V4] LOPO predictions on ALL rows (agreed + disagreed)")

    # Cut-MATCHED OLD label (current behaviour: same percentile as agreement)
    df_v4 = _make_old_combined_label(df, group_col, low_pctl, high_pctl, OLD_TARGET_COL)
    # Cut-UNIVERSAL OLD label (per-PID median split → every row labelled)
    df_v4 = _make_old_combined_label(df_v4, group_col, 0.50, 0.50, "__old_50_50__")
    # Cut-UNIVERSAL per-modality references
    df_v4 = df_v4.copy()
    df_v4["__tlx_50_50__"] = _per_pid_binary_signal(
        df_v4, group_col, TLX_SMOOTH_COL, 0.50, 0.50,
    )
    df_v4["__sleep_50_50__"] = _per_pid_binary_signal(
        df_v4, group_col, SLEEP_SMOOTH_COL, 0.50, 0.50,
    )
    # Proxy median-split (per-PID 0.50/0.50 of mean of proxy error cols)
    df_v4 = create_proxy_error_binary_target_per_participant(
        df=df_v4,
        group_col=group_col,
        proxy_cols=PROXY_ERROR_COLS,
        lower_percentile=0.50,
        upper_percentile=0.50,
        new_col="__proxy_50_50__",
    )

    cols = list(dict.fromkeys(
        selected_features + [
            hard_target_col, soft_target_col, weight_col, group_col,
            OLD_TARGET_COL, "__old_50_50__",
            TLX_SIGNAL_COL, SLEEP_SIGNAL_COL, PROXY_SIGNAL_COL,
            "__tlx_50_50__", "__sleep_50_50__", "__proxy_50_50__",
        ]
    ))
    sub = df_v4[cols].copy()
    # Keep all rows that have a participant; do NOT filter by label
    sub = sub[sub[group_col].notna()].reset_index(drop=True)

    X = sub[selected_features]
    numeric_features = [c for c in selected_features if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in selected_features if c not in numeric_features]
    groups = sub[group_col]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    y_hard = sub[hard_target_col].to_numpy(dtype=float)
    y_soft = sub[soft_target_col].to_numpy(dtype=float)
    weight = sub[weight_col].to_numpy(dtype=float)

    all_proba = np.full(len(sub), np.nan, dtype=float)
    logo = LeaveOneGroupOut()
    n_groups = int(groups.nunique())
    print(f"    LOPO over {n_groups} participants, predicting all rows")

    for fold_i, (tr, te) in enumerate(logo.split(X_scaled, np.zeros(len(sub)), groups), start=1):
        # Train Head A on training rows with hard labels
        tr_hard = tr[~np.isnan(y_hard[tr])]
        if len(tr_hard) < 10 or len(np.unique(y_hard[tr_hard])) < 2:
            continue
        sw = np.where(weight[tr_hard] > 0, weight[tr_hard], 1.0)
        pa = build_clf_pipeline(numeric_features, categorical_features)
        pa.fit(X_scaled.iloc[tr_hard], y_hard[tr_hard].astype(int),
               model__sample_weight=sw)
        proba_a = pa.predict_proba(X_scaled.iloc[te])[:, 1]

        # Train Head B on all training rows with soft target (broader coverage)
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

    # Sanity: report coverage
    coverage = np.mean(~np.isnan(all_proba))
    print(f"    Predictions generated for {int(np.sum(~np.isnan(all_proba)))}/{len(sub)} "
          f"rows ({coverage:.1%})")

    # ── Evaluate against multiple reference labels ─────────────────────────
    # The references are split into two camps:
    #   [cut-MATCHED]  — same percentile as agreement label. Rows in middle
    #                    band excluded. Coverage varies between runs.
    #   [UNIVERSAL]    — fixed 50/50 median split. Every row labelled.
    #                    Coverage identical across runs → comparable F1.
    references = [
        ("[agreement] hard label (where available)", sub[hard_target_col].to_numpy(dtype=float)),
        (f"[cut-MATCHED {low_pctl:.2f}/{high_pctl:.2f}] OLD combined TLX+sleepiness",
         sub[OLD_TARGET_COL].to_numpy(dtype=float)),
        (f"[cut-MATCHED {low_pctl:.2f}/{high_pctl:.2f}] tlx_signal",
         sub[TLX_SIGNAL_COL].to_numpy(dtype=float)),
        (f"[cut-MATCHED {low_pctl:.2f}/{high_pctl:.2f}] sleep_signal",
         sub[SLEEP_SIGNAL_COL].to_numpy(dtype=float)),
        (f"[cut-MATCHED {low_pctl:.2f}/{high_pctl:.2f}] proxy_signal",
         sub[PROXY_SIGNAL_COL].to_numpy(dtype=float)),
        ("[UNIVERSAL 50/50] OLD combined TLX+sleepiness",
         sub["__old_50_50__"].to_numpy(dtype=float)),
        ("[UNIVERSAL 50/50] tlx_signal",
         sub["__tlx_50_50__"].to_numpy(dtype=float)),
        ("[UNIVERSAL 50/50] sleep_signal",
         sub["__sleep_50_50__"].to_numpy(dtype=float)),
        ("[UNIVERSAL 50/50] proxy_signal",
         sub["__proxy_50_50__"].to_numpy(dtype=float)),
    ]
    rows = [_evaluate_predictions_against_label(all_proba, ref, name)
            for name, ref in references]

    # Sub-group breakdown vs the UNIVERSAL OLD-combined reference. Uses the
    # 50/50 reference so the row counts add up to the full 903 (modulo NaN
    # TLX/sleep). AGREED vs DISAGREED is defined by the agreement label.
    agreed_mask = ~np.isnan(y_hard)
    disagreed_mask = np.isnan(y_hard)
    universal_ref = sub["__old_50_50__"].to_numpy(dtype=float)
    for subset_name, m in [
        ("[UNIVERSAL 50/50] AGREED rows only (vs OLD combined)", agreed_mask),
        ("[UNIVERSAL 50/50] DISAGREED rows only (vs OLD combined)", disagreed_mask),
        ("[UNIVERSAL 50/50] ALL rows (vs OLD combined)",
         np.ones(len(sub), dtype=bool)),
    ]:
        proba_sub = np.where(m, all_proba, np.nan)
        ref_sub = np.where(m, universal_ref, np.nan)
        rows.append(_evaluate_predictions_against_label(proba_sub, ref_sub, subset_name))

    eval_df = pd.DataFrame(rows)

    print("\n  V4 results (out-of-fold LOPO predictions on all rows):")
    print("    NB: [cut-MATCHED] rows excludes middle-band; [UNIVERSAL] is comparable across cuts.")
    print(eval_df.round(3).to_string(index=False))

    # Build per-row prediction table for inspection
    pred_table = pd.DataFrame({
        "participant": groups.to_numpy(),
        "agreement_label": y_hard,
        f"old_combined_{low_pctl:.2f}_{high_pctl:.2f}": sub[OLD_TARGET_COL].to_numpy(dtype=float),
        f"tlx_signal_{low_pctl:.2f}_{high_pctl:.2f}": sub[TLX_SIGNAL_COL].to_numpy(dtype=float),
        f"sleep_signal_{low_pctl:.2f}_{high_pctl:.2f}": sub[SLEEP_SIGNAL_COL].to_numpy(dtype=float),
        f"proxy_signal_{low_pctl:.2f}_{high_pctl:.2f}": sub[PROXY_SIGNAL_COL].to_numpy(dtype=float),
        "old_combined_50_50": sub["__old_50_50__"].to_numpy(dtype=float),
        "tlx_signal_50_50": sub["__tlx_50_50__"].to_numpy(dtype=float),
        "sleep_signal_50_50": sub["__sleep_50_50__"].to_numpy(dtype=float),
        "proxy_signal_50_50": sub["__proxy_50_50__"].to_numpy(dtype=float),
        "lopo_proba": all_proba,
    })
    return eval_df, pred_table


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    low_pctl: float = LOW_PCTL,
    high_pctl: float = HIGH_PCTL,
    keep_ratio: float = 0.0,
    min_features: int = 25,
    precision_floor: float = 0.50,
    meta_top_k: int = 5,
    run_v1_dropped: bool = True,
    run_v2_inner_shap: bool = True,
    run_v3_old_label: bool = True,
    run_v4_all_rows: bool = True,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    print(f"Loaded data: shape {df.shape}")

    df = add_engineered_features(df)
    df = attach_smoothed_signals(df, group_col=GROUP_COL)
    print(f"After engineering + smoothing: shape {df.shape}")

    # Build agreement-denoised hard labels + confidence weights
    print("\n[Stage 1] Multi-modal agreement labeling")
    df, weights, audit = build_denoised_labels(df, group_col=GROUP_COL,
                                               low_pctl=low_pctl, high_pctl=high_pctl)
    audit.to_csv(LABEL_AUDIT_PATH, index=False)
    print(f"  Label audit (per (n_high, n_low) bucket):")
    print(audit.to_string(index=False))
    n_labeled = int(np.sum(~np.isnan(df[BINARY_TARGET_COL])))
    bal = pd.Series(df[BINARY_TARGET_COL]).value_counts(normalize=True).sort_index()
    print(f"  Total labelled rows: {n_labeled} / {len(df)}  (balance: {bal.to_dict()})")
    n_3of3 = int(np.sum(df[WEIGHT_COL] == 1.0))
    n_2of3_abstain = int(np.sum(df[WEIGHT_COL] == 0.7))
    n_2of3_disagree = int(np.sum(df[WEIGHT_COL] == 0.5))
    print(f"  Confidence: 3/3 agree = {n_3of3}, 2/3 abstain = {n_2of3_abstain}, "
          f"2/3 disagree = {n_2of3_disagree}")

    # Soft regression target
    df = attach_soft_target(df, group_col=GROUP_COL)

    # Feature universe
    #
    # LEAKAGE PREVENTION: PROXY_ERROR_COLS are the source of the
    # error_proxy_binary signal, which is one of the three votes in the
    # agreement label. If we left these columns in the feature set, the
    # model could trivially recover the proxy vote (just re-thresholding
    # the per-PID percentile of the mean of those 4 columns), and through
    # it ~1/3 of the agreement label. Strip them here.
    proxy_set = set(PROXY_ERROR_COLS)
    raw_features_full = FEATURE_GROUPS.get(TARGET_GROUP, [])
    raw_features = [f for f in raw_features_full if f not in proxy_set]
    n_removed = len(raw_features_full) - len(raw_features)
    extended_features = [f for f in (raw_features + ENGINEERED_FEATURE_NAMES)
                         if f in df.columns]
    print(f"  Excluded {n_removed} proxy-error columns from features "
          f"(leakage prevention): {sorted(proxy_set & set(raw_features_full))}")
    print(f"  Extended feature count: {len(extended_features)}")

    # SHAP feature selection on soft target
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

    # Within CV (two-head average)
    print("\n[Stage 3] Within-participant CV (two-head average: clf+reg)")
    wp_m, wp_yt, wp_ys, wp_gr = within_cv_two_head(
        df, selected, BINARY_TARGET_COL, SOFT_TARGET_COL, WEIGHT_COL, GROUP_COL,
    )
    _print_block("@ threshold 0.50", wp_m)
    wp_pr = _compute_pr_auc(wp_yt, wp_ys)
    print(f"    PR-AUC: {wp_pr:.3f}  baseline: "
          f"{np.mean(wp_yt) if wp_yt.size else float('nan'):.3f}")
    wp_g = best_threshold_global(wp_yt, wp_ys, precision_floor=precision_floor)
    print(f"  Global threshold-opt: t={wp_g['best_threshold']:.2f}  "
          f"f1={wp_g['f1']:.3f}  precision={wp_g['precision']:.3f}  recall={wp_g['recall']:.3f}")
    wp_pp = per_pid_threshold_metrics(wp_yt, wp_ys, wp_gr)
    _print_block("Per-pid threshold (IN-SAMPLE upper bound — not deployable)", wp_pp)

    # Cross CV (two-head stacked)
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

    # ─── Validation experiments ─────────────────────────────────────────────
    v1_table = None
    v2_metrics = None
    v2_pr = None
    v2_g = None
    v3_results = None

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
        v2_g = best_threshold_global(v2_yt, v2_ys, precision_floor=precision_floor)
        print(f"      PR-AUC={v2_pr:.3f}  global-t f1={v2_g['f1']:.3f} (t={v2_g['best_threshold']:.2f})")
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
                {"split": "cross",  **v3_results.get("cross", {})},
            ]).to_csv(VALIDATION_OLD_LABEL_PATH, index=False)

    v4_eval = None
    if run_v4_all_rows:
        v4_eval, v4_pred = validation_lopo_all_rows(
            df, selected, BINARY_TARGET_COL, SOFT_TARGET_COL, WEIGHT_COL, GROUP_COL,
            low_pctl=low_pctl, high_pctl=high_pctl,
        )
        v4_eval.to_csv(VALIDATION_ALL_ROWS_PATH, index=False)
        v4_pred.to_csv(VALIDATION_ALL_ROWS_PROBA_PATH, index=False)

    # ─── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY  (label-denoised + two-head stack + engineered features)")
    print("=" * 80)
    print("Headline result (agreement labels, single-pass SHAP):")
    print(f"  Within @ 0.50  :  f1={wp_m.get('f1', float('nan')):.3f}  PR-AUC={wp_pr:.3f}")
    print(f"  Within global  :  f1={wp_g['f1']:.3f}  (t={wp_g['best_threshold']:.2f})")
    print(f"  Within per-pid :  f1={wp_pp.get('f1', float('nan')):.3f}  (in-sample upper bound)")
    print(f"  Cross  @ 0.50  :  f1={cp_m.get('f1', float('nan')):.3f}  PR-AUC={cp_pr:.3f}")
    print(f"  Cross  global  :  f1={cp_g['f1']:.3f}  (t={cp_g['best_threshold']:.2f})")

    if v2_metrics is not None and v2_metrics:
        print("\nV2 — leakage-free (SHAP inside each LOPO fold):")
        print(f"  Cross  @ 0.50  :  f1={v2_metrics.get('f1', float('nan')):.3f}  PR-AUC={v2_pr:.3f}")
        print(f"  Cross  global  :  f1={v2_g['f1']:.3f}  (t={v2_g['best_threshold']:.2f})")
        delta_default = v2_metrics.get('f1', float('nan')) - cp_m.get('f1', float('nan'))
        print(f"  Δ vs headline  :  {delta_default:+.3f}  (negative = SHAP-on-full-data optimism)")

    if v3_results:
        v3_w = v3_results.get("within", {})
        v3_c = v3_results.get("cross", {})
        print(f"\nV3 — same {v3_results.get('n_intersection', '?')} rows but OLD label scheme "
              f"(combined TLX+sleepiness, {low_pctl:.2f}/{high_pctl:.2f}):")
        print(f"  Within @ 0.50  :  f1={v3_w.get('f1', float('nan')):.3f}  "
              f"PR-AUC={v3_w.get('pr_auc', float('nan')):.3f}")
        print(f"  Cross  @ 0.50  :  f1={v3_c.get('f1', float('nan')):.3f}  "
              f"PR-AUC={v3_c.get('pr_auc', float('nan')):.3f}")
        delta_within = v3_w.get('f1', float('nan')) - wp_m.get('f1', float('nan'))
        delta_cross = v3_c.get('f1', float('nan')) - cp_m.get('f1', float('nan'))
        print(f"  Δ within (OLD - agreement): {delta_within:+.3f}")
        print(f"  Δ cross  (OLD - agreement): {delta_cross:+.3f}")
        print(f"  Interpretation: large negative Δ → labeling protocol drives the gain;")
        print(f"                   ~0 Δ → the gain is the row-selection (easy cases).")

    if v1_table is not None:
        print(f"\nV1 — predictions on {int(v1_table.iloc[0]['n']) if len(v1_table) else 0} dropped rows:")
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
        # Pull the UNIVERSAL AGREED / DISAGREED / ALL lines for headline.
        # These use the fixed 50/50 OLD-combined reference, so identical
        # evaluable row sets across runs → safely comparable across cuts.
        focus_names = [
            "[UNIVERSAL 50/50] AGREED rows only (vs OLD combined)",
            "[UNIVERSAL 50/50] DISAGREED rows only (vs OLD combined)",
            "[UNIVERSAL 50/50] ALL rows (vs OLD combined)",
        ]
        focus = v4_eval[v4_eval["label"].isin(focus_names)]
        for _, row in focus.iterrows():
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
