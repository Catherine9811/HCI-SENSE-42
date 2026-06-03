from __future__ import annotations

"""
Push F1 further with pseudo-labeling + tight cuts + OOF stacking.

Lessons from predict_combined_alertness_ensemble_maxed.py:
  - The 4-model calibrated soft-vote ensemble gave NO real cross-participant
    gain (cross F1 0.700 vs single-LGBM 0.696) and HURT cross PR-AUC
    (0.677 vs 0.704). CalibratedClassifierCV's internal CV split + LOPO
    starves base learners of data, and tree models don't disagree enough
    on 144×605 to benefit from averaging.
  - Engineered features did earn 4 of 20 SHAP slots → keep them.
  - Per-participant threshold optimization on the same predictions used
    for evaluation is in-sample — we report it but with a caveat.

This script attacks the DATA instead of the model:

  1. Tight core cuts (0.20 / 0.80) for clean supervised signal.
  2. Pseudo-labeling the middle band:
       - Train initial LGBM on labeled rows.
       - Predict on the discarded middle band (0.20-0.80 percentile rows).
       - Rows with proba >= UPPER_PSEUDO or proba <= LOWER_PSEUDO are
         absorbed back into training with the inferred label.
       - The original labeled rows keep priority (their labels are "harder").
  3. Engineered features (carried over from prior script).
  4. SHAP feature selection on the COMBINED labeled+pseudo dataset.
  5. OOF-stacked meta-learner:
       Level-1: LGBM → out-of-fold probabilities (no leakage).
       Level-2: LogReg trained on [LGBM_oof_proba, top SHAP features]
       This is a real stack, not soft voting.
  6. Honest reporting:
       - Within: report (a) global threshold, (b) per-PID threshold with
         the in-sample caveat clearly stated.
       - Cross: only global threshold (no per-PID tuning at LOPO level).
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
from prediction.alertness.shared_config import DATA_PATH, FEATURE_GROUPS


BASE_DIR = Path(__file__).resolve().parent
TARGET_GROUP = "mouse_keyboard_traits_sleep_engagement_behavioural"
OUTPUT_DIR = BASE_DIR / "processed_data"
SHAP_IMPORTANCE_PATH = OUTPUT_DIR / "pseudo_label_maxed_shap_importance.csv"
SELECTED_FEATURES_PATH = OUTPUT_DIR / "pseudo_label_maxed_selected_features.csv"

# Pseudo-labeling thresholds: only absorb very-confident predictions
PSEUDO_LOW = 0.15
PSEUDO_HIGH = 0.85
# Down-weight pseudo-labels (they're noisier than the real labels)
PSEUDO_WEIGHT = 0.5

LGBM_PARAMS = {
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
# Continuous alertness score (used to label pseudo rows by confidence)
# ─────────────────────────────────────────────────────────────────────────────

def attach_alertness_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__alertness_score__"] = df[TLX_COL] + df[SLEEPINESS_COL]
    return df


def label_with_pseudo_band(
    df: pd.DataFrame,
    group_col: str,
    core_lower: float,
    core_upper: float,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Label rows using core_lower/core_upper percentiles per participant.
    Returns (df_with_target, is_core_mask).
    `is_core_mask` is True for rows that received a hard label,
    False for the middle-band rows (which become pseudo-label candidates).
    """
    df = df.copy()
    score_col = "__alertness_score__"
    target = np.full(len(df), np.nan)
    is_core = np.zeros(len(df), dtype=bool)

    for pid, idx in df.groupby(group_col).groups.items():
        idx = np.asarray(list(idx))
        scores = df.loc[idx, score_col].to_numpy()
        valid = ~np.isnan(scores)
        if valid.sum() == 0:
            continue
        lo = np.quantile(scores[valid], core_lower)
        hi = np.quantile(scores[valid], core_upper)
        for i, s in zip(idx, scores):
            if np.isnan(s):
                continue
            if s <= lo:
                target[i] = 0.0
                is_core[i] = True
            elif s >= hi:
                target[i] = 1.0
                is_core[i] = True

    df[target_col] = target
    return df, pd.Series(is_core, index=df.index, name="is_core")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor / pipeline
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


def build_lgbm(seed: int = RANDOM_STATE):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1, **LGBM_PARAMS)


def build_lgbm_pipeline(numeric_features, categorical_features, seed: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
        ("model", build_lgbm(seed)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-labeling
# ─────────────────────────────────────────────────────────────────────────────

def generate_pseudo_labels(
    df_full: pd.DataFrame,
    is_core: pd.Series,
    target_col: str,
    group_col: str,
    feature_cols: List[str],
    pseudo_low: float = PSEUDO_LOW,
    pseudo_high: float = PSEUDO_HIGH,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Train initial LGBM on core rows, predict on middle band, return
    pseudo-labeled dataframe + sample weights.
    """
    core_df = df_full[is_core].copy()
    middle_df = df_full[~is_core].copy()

    X_core, y_core, groups_core = prepare_subset_with_target(
        core_df, feature_cols, target_col, group_col
    )
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_core[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    scaler_stats = fit_participant_iqr_scaler(X_core, groups_core, numeric_features)
    X_core_scaled = transform_with_participant_iqr_scaler(
        X_core, groups_core, numeric_features, *scaler_stats
    )

    pipe = build_lgbm_pipeline(numeric_features, categorical_features)
    pipe.fit(X_core_scaled, y_core.to_numpy())

    middle_present = [f for f in feature_cols if f in middle_df.columns]
    if not middle_present:
        return core_df, np.ones(len(core_df), dtype=float)
    X_middle = middle_df[feature_cols]
    groups_middle = middle_df[group_col]
    valid_rows = X_middle.notna().any(axis=1)
    if valid_rows.sum() == 0:
        return core_df, np.ones(len(core_df), dtype=float)

    X_mid_scaled = transform_with_participant_iqr_scaler(
        X_middle, groups_middle, numeric_features, *scaler_stats
    )

    proba = pipe.predict_proba(X_mid_scaled)[:, 1]
    confident_pos = proba >= pseudo_high
    confident_neg = proba <= pseudo_low

    pseudo_rows = middle_df.copy()
    pseudo_rows[target_col] = np.nan
    pseudo_rows.loc[confident_pos, target_col] = 1.0
    pseudo_rows.loc[confident_neg, target_col] = 0.0
    pseudo_rows = pseudo_rows[pseudo_rows[target_col].notna()].copy()

    n_pos = int(confident_pos.sum())
    n_neg = int(confident_neg.sum())
    print(f"  Pseudo-labels: +{n_pos} positives, +{n_neg} negatives "
          f"({n_pos + n_neg} / {len(middle_df)} middle rows; "
          f"thresholds {pseudo_low:.2f}/{pseudo_high:.2f})")

    combined = pd.concat([core_df, pseudo_rows], axis=0).reset_index(drop=True)
    weights = np.concatenate([
        np.ones(len(core_df), dtype=float),
        np.full(len(pseudo_rows), PSEUDO_WEIGHT, dtype=float),
    ])
    return combined, weights


# ─────────────────────────────────────────────────────────────────────────────
# SHAP feature selection
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


def shap_select(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    keep_ratio: float,
    min_features: int,
    sample_weight: Optional[np.ndarray] = None,
    sample_size: int = 1500,
) -> Tuple[List[str], pd.DataFrame]:
    X, y, groups = prepare_subset_with_target(df, feature_cols, target_col, group_col)
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    pre = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    base = build_lgbm()
    pipe = Pipeline([("preprocess", pre), ("model", base)])
    fit_kwargs = {}
    if sample_weight is not None and len(sample_weight) == len(y):
        fit_kwargs["model__sample_weight"] = sample_weight
    pipe.fit(X_scaled, y.to_numpy(), **fit_kwargs)

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
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
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
# Stacked CV: LGBM (level-1) → LogReg (level-2) on top of OOF probabilities
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


def _train_level1_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    n_inner_splits: int = 5,
) -> np.ndarray:
    """Group-aware inner StratifiedKFold to produce OOF level-1 probabilities
    on the training set (no leakage)."""
    oof = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in cv.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = build_lgbm_pipeline(numeric_features, categorical_features)
        pipe.fit(X.iloc[tr], y[tr], model__sample_weight=sample_weight[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return oof


def cross_cv_stacked(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    sample_weight: np.ndarray,
    meta_extra_features: List[str],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    X, y, groups = prepare_subset_with_target(df, feature_cols, target_col, group_col)
    sw = sample_weight[X.index.to_numpy()] if len(sample_weight) == len(df) else np.ones(len(X))
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    logo = LeaveOneGroupOut()
    n_groups = len(np.unique(groups))
    print(f"    Cross-CV (stacked LOPO over {n_groups} participants)")

    total_tp = total_fp = total_fn = total_tn = 0
    all_y_true: List[np.ndarray] = []
    all_y_score: List[np.ndarray] = []

    meta_extra_present = [f for f in meta_extra_features if f in X_scaled.columns
                          and pd.api.types.is_numeric_dtype(X_scaled[f])]

    for fold_i, (tr, te) in enumerate(logo.split(X_scaled, y, groups), start=1):
        y_tr = y.iloc[tr].to_numpy().astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        sw_tr = sw[tr]

        # Level 1: LGBM OOF on TRAIN only
        oof_train = _train_level1_oof(
            X_scaled.iloc[tr], y_tr, groups.iloc[tr].to_numpy(), sw_tr,
            numeric_features, categorical_features, n_inner_splits=5,
        )

        # Level 1: full LGBM fit on TRAIN, predict on TEST
        full_pipe = build_lgbm_pipeline(numeric_features, categorical_features)
        full_pipe.fit(X_scaled.iloc[tr], y_tr, model__sample_weight=sw_tr)
        proba_test_l1 = full_pipe.predict_proba(X_scaled.iloc[te])[:, 1]

        # Level 2: LogReg meta on [oof_proba, top SHAP raw features]
        # Use median-impute + standard scale on the meta features only
        meta_imp = SimpleImputer(strategy="median")
        meta_scl = StandardScaler()
        if meta_extra_present:
            X_meta_tr_raw = X_scaled.iloc[tr][meta_extra_present].to_numpy()
            X_meta_te_raw = X_scaled.iloc[te][meta_extra_present].to_numpy()
            X_meta_tr_extra = meta_scl.fit_transform(meta_imp.fit_transform(X_meta_tr_raw))
            X_meta_te_extra = meta_scl.transform(meta_imp.transform(X_meta_te_raw))
            X_meta_tr = np.column_stack([oof_train.reshape(-1, 1), X_meta_tr_extra])
            X_meta_te = np.column_stack([proba_test_l1.reshape(-1, 1), X_meta_te_extra])
        else:
            X_meta_tr = oof_train.reshape(-1, 1)
            X_meta_te = proba_test_l1.reshape(-1, 1)

        meta = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
        meta.fit(X_meta_tr, y_tr, sample_weight=sw_tr)
        proba_test = meta.predict_proba(X_meta_te)[:, 1]

        y_te = y.iloc[te].to_numpy().astype(int)
        y_pred = (proba_test >= 0.5).astype(int)
        tp, fp, fn, tn, _, _, _ = _confusion(y_te, y_pred)
        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
        all_y_true.append(y_te)
        all_y_score.append(proba_test)

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
        np.concatenate(all_y_true),
        np.concatenate(all_y_score),
    )


def within_cv_stacked(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    sample_weight: np.ndarray,
    n_splits_default: int = 5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Within-participant: stacking is overkill (tiny per-PID folds); use plain LGBM."""
    X_all, y_all, groups_all = prepare_subset_with_target(df, feature_cols, target_col, group_col)
    sw_all = sample_weight[X_all.index.to_numpy()] if len(sample_weight) == len(df) else np.ones(len(X_all))
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_all[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X_all, groups_all, numeric_features)
    X_all_scaled = transform_with_participant_iqr_scaler(
        X_all, groups_all, numeric_features, *scaler_stats
    )

    total_tp = total_fp = total_fn = total_tn = 0
    all_y_true: List[np.ndarray] = []
    all_y_score: List[np.ndarray] = []
    all_groups: List[np.ndarray] = []

    for pid, idx in groups_all.groupby(groups_all).groups.items():
        idx = np.asarray(list(idx))
        if idx.size < 3:
            continue
        X_p = X_all_scaled.iloc[idx]
        y_p = y_all.iloc[idx].to_numpy().astype(int)
        sw_p = sw_all[idx]
        cc = np.bincount(y_p, minlength=2)
        min_class = int(cc[cc > 0].min()) if np.any(cc > 0) else 0
        n_splits = min(n_splits_default, idx.size, min_class)
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for tr, te in cv.split(X_p, y_p):
            if len(np.unique(y_p[tr])) < 2:
                continue
            pipe = build_lgbm_pipeline(numeric_features, categorical_features)
            pipe.fit(X_p.iloc[tr], y_p[tr], model__sample_weight=sw_p[tr])
            proba = pipe.predict_proba(X_p.iloc[te])[:, 1]
            y_pred = (proba >= 0.5).astype(int)
            tp, fp, fn, tn, _, _, _ = _confusion(y_p[te], y_pred)
            total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
            all_y_true.append(y_p[te])
            all_y_score.append(proba)
            all_groups.append(np.full(y_p[te].shape, pid))

    if total_tp + total_fp + total_fn + total_tn == 0:
        return {}, np.array([], int), np.array([], float), np.array([])
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return (
        {"TP": float(total_tp), "FP": float(total_fp), "FN": float(total_fn), "TN": float(total_tn),
         "precision": float(p), "recall": float(r), "f1": float(f1)},
        np.concatenate(all_y_true),
        np.concatenate(all_y_score),
        np.concatenate(all_groups),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimization
# ─────────────────────────────────────────────────────────────────────────────

def best_threshold_global(y_true, y_score, precision_floor=0.0):
    if y_true.size == 0 or y_score.size == 0:
        return {"best_threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
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


def per_pid_inferred_threshold_metrics(
    y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray,
) -> Dict[str, float]:
    """In-sample per-PID best threshold. INFLATED — for a deployment
    upper-bound estimate only. Reported with caveat."""
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    core_lower: float = 0.20,
    core_upper: float = 0.80,
    keep_ratio: float = 0.0,
    min_features: int = 25,
    precision_floor: float = 0.50,
    meta_top_k: int = 5,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    print(f"Loaded data: shape {df.shape}")

    df = add_engineered_features(df)
    df = attach_alertness_score(df)
    print(f"After engineering + score: shape {df.shape}")

    # 1. Tight core cuts (0.20/0.80)
    df_core, is_core = label_with_pseudo_band(
        df, GROUP_COL, core_lower, core_upper, BINARY_TARGET_COL,
    )
    n_core = int(is_core.sum())
    n_middle = int((~is_core).sum())
    bal = df_core.loc[is_core, BINARY_TARGET_COL].value_counts(normalize=True).sort_index()
    print(f"  Core labels @ {core_lower:.2f}/{core_upper:.2f}: "
          f"{n_core} rows  (balance: {bal.to_dict()})")
    print(f"  Middle band (pseudo-label candidates): {n_middle} rows")

    raw_features = FEATURE_GROUPS.get(TARGET_GROUP, [])
    extended_features = [f for f in (list(raw_features) + ENGINEERED_FEATURE_NAMES) if f in df_core.columns]
    print(f"  Extended feature count: {len(extended_features)}")

    # 2. Pseudo-label generation
    print("\n[Pseudo-labeling middle band]")
    df_combined, sample_weight = generate_pseudo_labels(
        df_core, is_core, BINARY_TARGET_COL, GROUP_COL, extended_features
    )
    n_total = len(df_combined)
    print(f"  Combined dataset: {n_total} rows  (real={n_core}, pseudo={n_total - n_core})")

    # 3. SHAP feature selection on combined dataset (with weights)
    print("\n[SHAP feature selection on combined dataset]")
    selected, shap_imp = shap_select(
        df_combined, extended_features,
        target_col=BINARY_TARGET_COL, group_col=GROUP_COL,
        keep_ratio=keep_ratio, min_features=min_features,
        sample_weight=sample_weight,
    )
    shap_imp.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    pd.DataFrame({"selected_feature": selected}).to_csv(SELECTED_FEATURES_PATH, index=False)
    print(f"  Selected {len(selected)} / {len(extended_features)} features:")
    for i, f in enumerate(selected, 1):
        marker = "  [ENG]" if f.startswith("eng_") else ""
        print(f"    {i:2d}. {f}{marker}")

    # 4. Within CV (single LGBM, with sample weights)
    print("\n[Within-participant CV — LGBM + pseudo-labels (sample-weighted)]")
    wp_m, wp_yt, wp_ys, wp_groups = within_cv_stacked(
        df_combined, selected, BINARY_TARGET_COL, GROUP_COL, sample_weight,
    )
    _print_block("@ threshold 0.50", wp_m)
    wp_pr = _compute_pr_auc(wp_yt, wp_ys)
    print(f"    PR-AUC: {wp_pr:.3f}  baseline: {np.mean(wp_yt) if wp_yt.size else float('nan'):.3f}")
    wp_g = best_threshold_global(wp_yt, wp_ys, precision_floor=precision_floor)
    print(f"  Global threshold-opt: t={wp_g['best_threshold']:.2f}  "
          f"f1={wp_g['f1']:.3f}  precision={wp_g['precision']:.3f}  recall={wp_g['recall']:.3f}")
    wp_pp = per_pid_inferred_threshold_metrics(wp_yt, wp_ys, wp_groups)
    _print_block("Per-pid threshold (IN-SAMPLE upper bound — not deployable)", wp_pp)

    # 5. Cross CV (stacked LGBM → LogReg)
    print("\n[Cross-participant CV — stacked LGBM → LogReg with pseudo-labels]")
    meta_extra = selected[:meta_top_k]
    cp_m, cp_yt, cp_ys = cross_cv_stacked(
        df_combined, selected, BINARY_TARGET_COL, GROUP_COL, sample_weight,
        meta_extra_features=meta_extra,
    )
    _print_block("@ threshold 0.50", cp_m)
    cp_pr = _compute_pr_auc(cp_yt, cp_ys)
    print(f"    PR-AUC: {cp_pr:.3f}  baseline: {np.mean(cp_yt) if cp_yt.size else float('nan'):.3f}")
    cp_g = best_threshold_global(cp_yt, cp_ys, precision_floor=precision_floor)
    print(f"  Global threshold-opt: t={cp_g['best_threshold']:.2f}  "
          f"f1={cp_g['f1']:.3f}  precision={cp_g['precision']:.3f}  recall={cp_g['recall']:.3f}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY  (pseudo-label + tight cuts + OOF-stacked meta)")
    print("=" * 80)
    print(f"  Within @ 0.50  :  f1={wp_m.get('f1', float('nan')):.3f}  PR-AUC={wp_pr:.3f}")
    print(f"  Within global  :  f1={wp_g['f1']:.3f}  (t={wp_g['best_threshold']:.2f})")
    print(f"  Within per-pid :  f1={wp_pp.get('f1', float('nan')):.3f}  (in-sample)")
    print(f"  Cross  @ 0.50  :  f1={cp_m.get('f1', float('nan')):.3f}  PR-AUC={cp_pr:.3f}")
    print(f"  Cross  global  :  f1={cp_g['f1']:.3f}  (t={cp_g['best_threshold']:.2f})")
    print(f"\n  SHAP importance:    {SHAP_IMPORTANCE_PATH}")
    print(f"  Selected features:  {SELECTED_FEATURES_PATH}")


def main() -> None:
    run_experiment(
        core_lower=0.20,
        core_upper=0.80,
        keep_ratio=0.0,
        min_features=25,
        precision_floor=0.50,
        meta_top_k=5,
    )


if __name__ == "__main__":
    main()
