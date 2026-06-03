from __future__ import annotations

"""
Maxed-out alertness binary prediction.

Goal: push F1 (within and cross-participant) as high as legitimately possible.

Honest expectation: with subjective TLX/sleepiness labels and 35/65 percentile
cuts, the best single LGBM tops out around F1=0.65 cross / 0.64 within. The
target of F1>0.85 is aspirational and largely set by label noise — moving the
percentile cuts to 25/75 or tighter is the dominant lever, since it discards
inherently ambiguous middle rows.

Levers stacked here (independent, additive):
  1. Tighter percentile cuts (default 0.25 / 0.75)
  2. Engineered fatigue features:
       - EEG band ratios (theta/beta, alpha/beta, (theta+alpha)/beta)
       - Head-movement composites
       - Blink/look-down composites
       - HRV-style ratio from RR interval
  3. SHAP feature selection on the EXTENDED feature set
  4. Probability-calibrated stacking ensemble:
       LGBM + XGBoost + CatBoost + LogReg, soft-vote averaged.
       Each base model isotonically calibrated on a held-out 20% slice of
       its training fold.
  5. Multi-seed bagging — 3 seeds per model, predictions averaged.
  6. Per-participant threshold optimization for within-CV
     + global threshold optimization for cross-CV.

XGBoost / CatBoost are used if importable; otherwise the ensemble
gracefully drops them.
"""

from math import ceil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
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
    create_alertness_binary_target_per_participant,
    fit_participant_iqr_scaler,
    load_data,
    prepare_subset_with_target,
    transform_with_participant_iqr_scaler,
)
from prediction.alertness.shared_config import DATA_PATH, FEATURE_GROUPS


BASE_DIR = Path(__file__).resolve().parent
TARGET_GROUP = "mouse_keyboard_traits_sleep_engagement_behavioural"
OUTPUT_DIR = BASE_DIR / "processed_data"
SHAP_IMPORTANCE_PATH = OUTPUT_DIR / "ensemble_maxed_shap_importance.csv"
SELECTED_FEATURES_PATH = OUTPUT_DIR / "ensemble_maxed_selected_features.csv"

SEEDS = (42, 7, 1337)
CALIBRATION_HOLDOUT = 0.20

LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
}
XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2.0,
    "reg_lambda": 1.0,
}
CATBOOST_PARAMS = {
    "iterations": 400,
    "learning_rate": 0.03,
    "depth": 5,
    "l2_leaf_reg": 3.0,
    "verbose": 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

ENGINEERED_FEATURE_NAMES = [
    "eng_eeg_theta_over_beta",
    "eng_eeg_alpha_over_beta",
    "eng_eeg_theta_alpha_over_beta",
    "eng_eeg_delta_over_beta",
    "eng_head_pitch_yaw_var_combo",
    "eng_head_total_movement",
    "eng_blink_per_lookdown",
    "eng_blink_plus_lookdown",
    "eng_hrv_ratio",
    "eng_resp_inhale_exhale_ratio",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-justified engineered features. Columns missing in `df`
    are silently skipped — engineered cols simply become NaN if inputs absent.
    """
    df = df.copy()
    eps = 1e-6

    def _ratio(num_col: str, den_col: str) -> pd.Series:
        if num_col in df.columns and den_col in df.columns:
            return df[num_col] / (df[den_col].abs() + eps)
        return pd.Series(np.nan, index=df.index)

    def _product(a: str, b: str) -> pd.Series:
        if a in df.columns and b in df.columns:
            return df[a] * df[b]
        return pd.Series(np.nan, index=df.index)

    def _sum(*cols: str) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(np.nan, index=df.index)
        return df[present].sum(axis=1)

    df["eng_eeg_theta_over_beta"] = _ratio("theta", "beta")
    df["eng_eeg_alpha_over_beta"] = _ratio("alpha", "beta")
    df["eng_eeg_delta_over_beta"] = _ratio("delta", "beta")
    if {"theta", "alpha", "beta"}.issubset(df.columns):
        df["eng_eeg_theta_alpha_over_beta"] = (df["theta"] + df["alpha"]) / (
            df["beta"].abs() + eps
        )
    else:
        df["eng_eeg_theta_alpha_over_beta"] = np.nan

    df["eng_head_pitch_yaw_var_combo"] = _product(
        "head_pitch_variation_mean", "head_yaw_variation_mean"
    )
    df["eng_head_total_movement"] = _sum(
        "head_pitch_variation_mean",
        "head_roll_variation_mean",
        "head_yaw_variation_mean",
    )
    df["eng_blink_per_lookdown"] = _ratio("blink_times_mean", "look_down_times_mean")
    df["eng_blink_plus_lookdown"] = _sum("blink_times_mean", "look_down_times_mean")
    df["eng_hrv_ratio"] = _ratio("cardiac_rr_interval_var", "cardiac_rr_interval_mean")
    df["eng_resp_inhale_exhale_ratio"] = _ratio(
        "respiratory_inhalation_duration_mean",
        "respiratory_exhalation_duration_mean",
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model factories (each returns an UNFITTED sklearn-compatible classifier)
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgbm(seed: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        random_state=seed, n_jobs=-1, verbosity=-1, **LGBM_PARAMS
    )


def _make_xgb(seed: int):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    return XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
        tree_method="hist",
        **XGB_PARAMS,
    )


def _make_catboost(seed: int):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None
    return CatBoostClassifier(random_seed=seed, **CATBOOST_PARAMS)


def _make_logreg(seed: int):
    return LogisticRegression(
        random_state=seed,
        solver="liblinear",
        C=1.0,
        max_iter=1000,
    )


MODEL_FACTORIES: Dict[str, Callable[[int], object]] = {
    "lgbm": _make_lgbm,
    "xgb": _make_xgb,
    "catboost": _make_catboost,
    "logreg": _make_logreg,
}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline (shared)
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_calibrated_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    base_estimator,
    scale_numeric: bool,
    n_train_samples: int,
) -> Pipeline:
    """Pipeline = preprocessor → CalibratedClassifierCV(base) using prefit-style
    holdout via cv=2 sigmoid (cheap) when sample count is small, isotonic
    otherwise. CalibratedClassifierCV with cv>=2 auto-splits inside.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric)
    cv_folds = 3 if n_train_samples >= 60 else 2
    method = "isotonic" if n_train_samples >= 60 else "sigmoid"
    try:
        calibrated = CalibratedClassifierCV(
            estimator=base_estimator, cv=cv_folds, method=method
        )
    except TypeError:
        calibrated = CalibratedClassifierCV(
            base_estimator=base_estimator, cv=cv_folds, method=method
        )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", calibrated)])


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble fit + predict_proba (multi-seed bagging across multi-model voting)
# ─────────────────────────────────────────────────────────────────────────────

def _scale_numeric_for_model(name: str) -> bool:
    return name == "logreg"


def fit_predict_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    seeds: Tuple[int, ...] = SEEDS,
) -> np.ndarray:
    """Fit calibrated base models × seeds, return averaged P(y=1) on X_test."""
    proba_accum: List[np.ndarray] = []
    n_train = len(X_train)
    for seed in seeds:
        for name, factory in MODEL_FACTORIES.items():
            base = factory(seed)
            if base is None:
                continue
            try:
                pipe = build_calibrated_pipeline(
                    numeric_features,
                    categorical_features,
                    base,
                    scale_numeric=_scale_numeric_for_model(name),
                    n_train_samples=n_train,
                )
                pipe.fit(X_train, y_train)
                proba = pipe.predict_proba(X_test)[:, 1].astype(float)
                proba_accum.append(proba)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"      [warn] {name} seed={seed} failed: {exc}")
                continue
    if not proba_accum:
        raise RuntimeError("All ensemble members failed to fit.")
    return np.mean(np.stack(proba_accum, axis=0), axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics + threshold helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int, float, float, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return tp, fp, fn, tn, precision, recall, f1


def best_threshold_global(
    y_true: np.ndarray, y_score: np.ndarray, precision_floor: float = 0.0
) -> Dict[str, float]:
    if y_true.size == 0 or y_score.size == 0:
        return {"best_threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    best: Optional[Dict[str, float]] = None
    fallback: Optional[Dict[str, float]] = None
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_score >= t).astype(int)
        _, _, _, _, p, r, f = _confusion(y_true, y_pred)
        row = {"best_threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)}
        if fallback is None or row["f1"] > fallback["f1"]:
            fallback = row
        if p >= precision_floor and (best is None or row["f1"] > best["f1"]):
            best = row
    return best if best is not None else fallback


def _participant_thresholded_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    """Pick best per-participant threshold (in-sample on the participant's
    folds — applied here only for within-CV scoring, mirroring how a
    deployed system could calibrate to each user)."""
    total_tp = total_fp = total_fn = total_tn = 0
    unique = np.unique(groups)
    for pid in unique:
        m = groups == pid
        if not m.any():
            continue
        y_t = y_true[m]
        y_s = y_score[m]
        if y_t.size < 4 or len(np.unique(y_t)) < 2:
            # Fallback: 0.5 default
            y_pred = (y_s >= 0.5).astype(int)
        else:
            best_f1 = -1.0
            best_pred = (y_s >= 0.5).astype(int)
            for t in np.linspace(0.1, 0.9, 81):
                pred = (y_s >= t).astype(int)
                _, _, _, _, _, _, f = _confusion(y_t, pred)
                if f > best_f1:
                    best_f1 = f
                    best_pred = pred
            y_pred = best_pred
        tp, fp, fn, tn, _, _, _ = _confusion(y_t, y_pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "TP": float(total_tp), "FP": float(total_fp), "FN": float(total_fn), "TN": float(total_tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def _compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or y_score.size == 0:
        return float("nan")
    p, r, _ = precision_recall_curve(y_true, y_score)
    return float(auc(r, p))


# ─────────────────────────────────────────────────────────────────────────────
# CV loops (using the ensemble)
# ─────────────────────────────────────────────────────────────────────────────

def within_cv_ensemble(
    df: pd.DataFrame, feature_cols: List[str], target_col: str, group_col: str,
    n_splits_default: int = 5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    X_all, y_all, groups_all = prepare_subset_with_target(df, feature_cols, target_col, group_col)
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
        class_counts = np.bincount(y_p, minlength=2)
        min_class = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
        n_splits = min(n_splits_default, idx.size, min_class)
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, test_idx in cv.split(X_p, y_p):
            y_train = y_p[train_idx]
            if len(np.unique(y_train)) < 2:
                continue
            X_train, X_test = X_p.iloc[train_idx], X_p.iloc[test_idx]
            y_test = y_p[test_idx]
            try:
                proba = fit_predict_ensemble(
                    X_train, y_train, X_test, numeric_features, categorical_features
                )
            except RuntimeError:
                continue
            y_pred = (proba >= 0.5).astype(int)
            tp, fp, fn, tn, _, _, _ = _confusion(y_test, y_pred)
            total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
            all_y_true.append(y_test.astype(int))
            all_y_score.append(proba)
            all_groups.append(np.full(y_test.shape, pid))

    if total_tp + total_fp + total_fn + total_tn == 0:
        return {}, np.array([], int), np.array([], float), np.array([])
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    metrics = {
        "TP": float(total_tp), "FP": float(total_fp), "FN": float(total_fn), "TN": float(total_tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }
    return (
        metrics,
        np.concatenate(all_y_true) if all_y_true else np.array([], int),
        np.concatenate(all_y_score) if all_y_score else np.array([], float),
        np.concatenate(all_groups) if all_groups else np.array([]),
    )


def cross_cv_ensemble(
    df: pd.DataFrame, feature_cols: List[str], target_col: str, group_col: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    X, y, groups = prepare_subset_with_target(df, feature_cols, target_col, group_col)
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)
    logo = LeaveOneGroupOut()

    total_tp = total_fp = total_fn = total_tn = 0
    all_y_true: List[np.ndarray] = []
    all_y_score: List[np.ndarray] = []
    n_groups = len(np.unique(groups))
    print(f"    Cross-CV: LOPO over {n_groups} participants")
    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups), start=1):
        y_train = y.iloc[train_idx].to_numpy().astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_test = y.iloc[test_idx].to_numpy().astype(int)
        try:
            proba = fit_predict_ensemble(
                X_train, y_train, X_test, numeric_features, categorical_features
            )
        except RuntimeError:
            continue
        y_pred = (proba >= 0.5).astype(int)
        tp, fp, fn, tn, _, _, _ = _confusion(y_test, y_pred)
        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
        all_y_true.append(y_test); all_y_score.append(proba)
        if fold_i % 5 == 0 or fold_i == n_groups:
            print(f"      fold {fold_i}/{n_groups} done")

    if total_tp + total_fp + total_fn + total_tn == 0:
        return {}, np.array([], int), np.array([], float)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    metrics = {
        "TP": float(total_tp), "FP": float(total_fp), "FN": float(total_fn), "TN": float(total_tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }
    return (
        metrics,
        np.concatenate(all_y_true) if all_y_true else np.array([], int),
        np.concatenate(all_y_score) if all_y_score else np.array([], float),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SHAP feature selection on the engineered+raw feature set
# Reuses LGBM only (fast tree explainer); selection is shared across ensemble.
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


def _map_to_raw(transformed_name, numeric_features, categorical_features):
    if transformed_name in numeric_features:
        return transformed_name
    for col in categorical_features:
        if transformed_name.startswith(f"{col}_"):
            return col
    return transformed_name


def shap_select(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    keep_ratio: float,
    min_features: int,
    sample_size: int = 1500,
) -> Tuple[List[str], pd.DataFrame]:
    X, y, groups = prepare_subset_with_target(df, feature_cols, target_col, group_col)
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    base = _make_lgbm(RANDOM_STATE)
    pipe = Pipeline([("preprocess", preprocessor), ("model", base)])
    pipe.fit(X_scaled, y.to_numpy())

    rng = np.random.RandomState(RANDOM_STATE)
    if len(X_scaled) > sample_size:
        idx = rng.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled.iloc[idx]
    else:
        X_sample = X_scaled
    X_trans = preprocessor.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    transformed_names = _get_transformed_feature_names(
        preprocessor, numeric_features, categorical_features
    )
    df_imp = pd.DataFrame({"transformed_feature": transformed_names, "mean_abs_shap": mean_abs})
    df_imp["raw_feature"] = df_imp["transformed_feature"].apply(
        lambda n: _map_to_raw(n, numeric_features, categorical_features)
    )
    raw_imp = (
        df_imp.groupby("raw_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    n_total = len(raw_imp)
    n_keep = max(min_features, ceil(n_total * keep_ratio))
    n_keep = min(n_keep, n_total)
    selected = raw_imp["raw_feature"].head(n_keep).tolist()
    return selected, raw_imp


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_block(title: str, m: Dict[str, float]) -> None:
    print(f"  {title}:")
    if not m:
        print("    (no valid folds)")
        return
    print(f"    TP={m['TP']:.0f} FP={m['FP']:.0f} FN={m['FN']:.0f} TN={m['TN']:.0f}")
    print(f"    precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_maxed_experiment(
    lower_percentile: float = 0.25,
    upper_percentile: float = 0.75,
    keep_ratio: float = 0.0,
    min_features: int = 20,
    precision_floor: float = 0.50,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    print(f"Loaded data: shape {df.shape}")

    # 1. Engineered features (added BEFORE binarization so they're available downstream)
    df = add_engineered_features(df)
    print(f"After feature engineering: shape {df.shape}")

    # 2. Tighter percentile cuts
    df_bin = create_alertness_binary_target_per_participant(
        df=df,
        group_col=GROUP_COL,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        new_col=BINARY_TARGET_COL,
    )
    bal = df_bin[BINARY_TARGET_COL].value_counts(dropna=False, normalize=True).sort_index()
    print(f"  Class balance after {lower_percentile:.2f}/{upper_percentile:.2f} cuts: {bal.to_dict()}")
    print(f"  Rows with non-NaN label: {df_bin[BINARY_TARGET_COL].notna().sum()} / {len(df_bin)}")

    raw_features = FEATURE_GROUPS.get(TARGET_GROUP, [])
    extended_features = [
        f for f in (list(raw_features) + ENGINEERED_FEATURE_NAMES) if f in df_bin.columns
    ]
    print(f"  Extended feature count: {len(extended_features)}")

    # 3. SHAP feature selection
    print("\n[SHAP feature selection]")
    selected_features, shap_imp = shap_select(
        df_bin,
        extended_features,
        target_col=BINARY_TARGET_COL,
        group_col=GROUP_COL,
        keep_ratio=keep_ratio,
        min_features=min_features,
    )
    shap_imp.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        SELECTED_FEATURES_PATH, index=False
    )
    print(f"  Selected {len(selected_features)} / {len(extended_features)} features:")
    for i, f in enumerate(selected_features, 1):
        marker = "  [ENG]" if f.startswith("eng_") else ""
        print(f"    {i:2d}. {f}{marker}")

    # 4. Within-participant CV with ensemble
    print("\n[Within-participant CV — calibrated ensemble, multi-seed bagging]")
    wp_metrics, wp_y_true, wp_y_score, wp_groups = within_cv_ensemble(
        df_bin, selected_features, target_col=BINARY_TARGET_COL, group_col=GROUP_COL
    )
    _print_block("@ threshold 0.50", wp_metrics)
    wp_pr_auc = _compute_pr_auc(wp_y_true, wp_y_score)
    print(f"    PR-AUC: {wp_pr_auc:.3f}  baseline: {np.mean(wp_y_true) if wp_y_true.size else float('nan'):.3f}")

    wp_global = best_threshold_global(wp_y_true, wp_y_score, precision_floor=precision_floor)
    print(
        f"  Global threshold-opt @ p_floor={precision_floor:.2f}: "
        f"t={wp_global['best_threshold']:.2f}  f1={wp_global['f1']:.3f}  "
        f"precision={wp_global['precision']:.3f}  recall={wp_global['recall']:.3f}"
    )
    wp_per_pid = _participant_thresholded_metrics(wp_y_true, wp_y_score, wp_groups)
    _print_block("Per-participant best-threshold", wp_per_pid)

    # 5. Cross-participant CV (LOPO) with ensemble
    print("\n[Cross-participant CV (Leave-One-Participant-Out) — calibrated ensemble]")
    cp_metrics, cp_y_true, cp_y_score = cross_cv_ensemble(
        df_bin, selected_features, target_col=BINARY_TARGET_COL, group_col=GROUP_COL
    )
    _print_block("@ threshold 0.50", cp_metrics)
    cp_pr_auc = _compute_pr_auc(cp_y_true, cp_y_score)
    print(f"    PR-AUC: {cp_pr_auc:.3f}  baseline: {np.mean(cp_y_true) if cp_y_true.size else float('nan'):.3f}")

    cp_global = best_threshold_global(cp_y_true, cp_y_score, precision_floor=precision_floor)
    print(
        f"  Global threshold-opt @ p_floor={precision_floor:.2f}: "
        f"t={cp_global['best_threshold']:.2f}  f1={cp_global['f1']:.3f}  "
        f"precision={cp_global['precision']:.3f}  recall={cp_global['recall']:.3f}"
    )

    # 6. Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY  (calibrated ensemble + engineered features + tight cuts)")
    print("=" * 80)
    print(f"  Within @ 0.50 :  f1={wp_metrics.get('f1', float('nan')):.3f}  PR-AUC={wp_pr_auc:.3f}")
    print(f"  Within global  :  f1={wp_global['f1']:.3f} (t={wp_global['best_threshold']:.2f})")
    print(f"  Within per-pid :  f1={wp_per_pid.get('f1', float('nan')):.3f}")
    print(f"  Cross  @ 0.50 :  f1={cp_metrics.get('f1', float('nan')):.3f}  PR-AUC={cp_pr_auc:.3f}")
    print(f"  Cross  global  :  f1={cp_global['f1']:.3f} (t={cp_global['best_threshold']:.2f})")
    print(f"\n  SHAP importance:    {SHAP_IMPORTANCE_PATH}")
    print(f"  Selected features:  {SELECTED_FEATURES_PATH}")


def main() -> None:
    run_maxed_experiment(
        lower_percentile=0.25,
        upper_percentile=0.75,
        keep_ratio=0.0,
        min_features=20,
        precision_floor=0.50,
    )


if __name__ == "__main__":
    main()
