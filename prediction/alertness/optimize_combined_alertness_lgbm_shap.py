from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
from prediction.alertness.predict_combined_alertness_lgbm_binary_participant_iqr_scaled_shap_select import (
    TARGET_GROUP,
    compute_shap_raw_feature_importance,
)
from prediction.alertness.shared_config import DATA_PATH, FEATURE_GROUPS


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "processed_data" / "combined_alertness_lgbm_iqr_optimization"


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or y_score.size == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def compute_pr_baseline(y_true: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true))


def _get_lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise ImportError(
            "Failed to import LightGBM. Please ensure compatible versions of "
            "lightgbm/pandas/dask are installed in the current environment."
        ) from exc
    return LGBMClassifier


def build_lgbm_pipeline_with_params(
    numeric_features: List[str],
    categorical_features: List[str],
    lgbm_params: Dict[str, float],
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    LGBMClassifier = _get_lgbm_classifier()
    model = LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **lgbm_params,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def select_features_by_shap(
    importance_df: pd.DataFrame,
    min_features: int,
) -> List[str]:
    if importance_df.empty:
        return []
    n_keep = min(max(1, min_features), len(importance_df))
    return importance_df["raw_feature"].head(n_keep).tolist()


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    precision_floor: float = 0.0,
    beta: float = 1.0,
    t_min: float = 0.05,
    t_max: float = 0.95,
    t_steps: int = 181,
) -> Dict[str, float]:
    if y_true.size == 0 or y_score.size == 0:
        return {
            "best_threshold": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "TP": float("nan"),
            "FP": float("nan"),
            "FN": float("nan"),
            "TN": float("nan"),
        }

    thresholds = np.linspace(t_min, t_max, t_steps)
    best_feasible = None
    best_any = None
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)
        f_beta = _fbeta(precision, recall, beta)
        item = {
            "best_threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f_beta": float(f_beta),
            "TP": float(tp),
            "FP": float(fp),
            "FN": float(fn),
            "TN": float(tn),
        }
        if best_any is None or item["f_beta"] > best_any["f_beta"]:
            best_any = item
        if precision >= precision_floor:
            if best_feasible is None or item["f_beta"] > best_feasible["f_beta"]:
                best_feasible = item
    return best_feasible if best_feasible is not None else (best_any if best_any is not None else {})


def _compute_confusion_and_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int, float, float, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)
    return tp, fp, fn, tn, precision, recall, f1


def within_participant_cv_binary_lgbm_with_params(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    lgbm_params: Dict[str, float],
    n_splits_default: int = 5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
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
    groups_dict = groups_all.groupby(groups_all).groups

    for _, idx in groups_dict.items():
        idx = np.asarray(list(idx))
        if idx.size < 3:
            continue
        n_splits = min(n_splits_default, idx.size)
        if n_splits < 2:
            continue
        X_p = X_all_scaled.iloc[idx]
        y_p = y_all.iloc[idx].to_numpy().astype(int)
        class_counts = np.bincount(y_p, minlength=2)
        min_class_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
        n_splits = min(n_splits, min_class_count)
        if n_splits < 2:
            continue

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, test_idx in cv.split(X_p, y_p):
            y_train = y_p[train_idx]
            if len(np.unique(y_train)) < 2:
                continue
            model = build_lgbm_pipeline_with_params(
                numeric_features, categorical_features, lgbm_params
            )
            X_train, X_test = X_p.iloc[train_idx], X_p.iloc[test_idx]
            y_test = y_p[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).astype(int)
            y_score = model.predict_proba(X_test)[:, 1].astype(float)
            tp, fp, fn, tn, _, _, _ = _compute_confusion_and_scores(y_test, y_pred)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            all_y_true.append(y_test.astype(int))
            all_y_score.append(y_score)

    total = total_tp + total_fp + total_fn + total_tn
    if total == 0:
        return {}, np.array([], dtype=int), np.array([], dtype=float)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)
    metrics = {
        "TP": float(total_tp),
        "FP": float(total_fp),
        "FN": float(total_fn),
        "TN": float(total_tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    y_true_all = np.concatenate(all_y_true) if all_y_true else np.array([], dtype=int)
    y_score_all = np.concatenate(all_y_score) if all_y_score else np.array([], dtype=float)
    return metrics, y_true_all, y_score_all


def cross_participant_cv_binary_lgbm_with_params(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    lgbm_params: Dict[str, float],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    X, y, groups = prepare_subset_with_target(df, feature_cols, target_col, group_col)
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    total_tp = total_fp = total_fn = total_tn = 0
    all_y_true: List[np.ndarray] = []
    all_y_score: List[np.ndarray] = []
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        y_train = y.iloc[train_idx].to_numpy().astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        model = build_lgbm_pipeline_with_params(
            numeric_features, categorical_features, lgbm_params
        )
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_test = y.iloc[test_idx].to_numpy().astype(int)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).astype(int)
        y_score = model.predict_proba(X_test)[:, 1].astype(float)
        tp, fp, fn, tn, _, _, _ = _compute_confusion_and_scores(y_test, y_pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        all_y_true.append(y_test)
        all_y_score.append(y_score)

    total = total_tp + total_fp + total_fn + total_tn
    if total == 0:
        return {}, np.array([], dtype=int), np.array([], dtype=float)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)
    metrics = {
        "TP": float(total_tp),
        "FP": float(total_fp),
        "FN": float(total_fn),
        "TN": float(total_tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    y_true_all = np.concatenate(all_y_true) if all_y_true else np.array([], dtype=int)
    y_score_all = np.concatenate(all_y_score) if all_y_score else np.array([], dtype=float)
    return metrics, y_true_all, y_score_all


def sample_lgbm_param_candidates(
    n_candidates: int = 80,
    seed: int = RANDOM_STATE,
) -> List[Dict[str, float]]:
    rng = np.random.RandomState(seed)
    candidates: List[Dict[str, float]] = []
    seen = set()
    while len(candidates) < n_candidates:
        params = {
            "n_estimators": int(rng.choice([200, 300, 400, 500, 700])),
            "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.07, 0.10])),
            "num_leaves": int(rng.choice([7, 11, 15, 23, 31, 47])),
            "min_child_samples": int(rng.choice([2, 5, 10, 20, 30])),
            "min_split_gain": float(rng.choice([0.0, 0.01, 0.05, 0.1])),
            "subsample": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
            "colsample_bytree": float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
            "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0, 2.0])),
            "reg_lambda": float(rng.choice([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])),
        }
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(params)
    return candidates


def run_once(
    df_raw: pd.DataFrame,
    lower: float,
    upper: float,
    keep_ratio: float,
    min_features: int,
    precision_floor: float,
    beta: float,
    lgbm_params: Dict[str, float],
) -> Dict[str, float]:
    df_bin = create_alertness_binary_target_per_participant(
        df=df_raw.copy(),
        group_col=GROUP_COL,
        lower_percentile=lower,
        upper_percentile=upper,
        new_col=BINARY_TARGET_COL,
    )
    raw_features = FEATURE_GROUPS.get(TARGET_GROUP, [])
    existing_features = [f for f in raw_features if f in df_bin.columns]
    if not existing_features:
        return {}

    shap_importance = compute_shap_raw_feature_importance(
        df=df_bin,
        feature_cols=existing_features,
        target_col=BINARY_TARGET_COL,
        group_col=GROUP_COL,
    )
    selected_features = select_features_by_shap(
        shap_importance,
        min_features=min_features,
    )
    if not selected_features:
        return {}

    wp_results, wp_y_true, wp_y_score = within_participant_cv_binary_lgbm_with_params(
        df_bin,
        selected_features,
        target_col=BINARY_TARGET_COL,
        group_col=GROUP_COL,
        lgbm_params=lgbm_params,
    )
    cp_results, cp_y_true, cp_y_score = cross_participant_cv_binary_lgbm_with_params(
        df_bin,
        selected_features,
        target_col=BINARY_TARGET_COL,
        group_col=GROUP_COL,
        lgbm_params=lgbm_params,
    )
    if (not wp_results) or (not cp_results):
        return {}

    wp_opt = threshold_sweep(
        wp_y_true,
        wp_y_score,
        precision_floor=precision_floor,
        beta=beta,
    )
    cp_opt = threshold_sweep(
        cp_y_true,
        cp_y_score,
        precision_floor=precision_floor,
        beta=beta,
    )

    wp_pr_auc = compute_pr_auc(wp_y_true, wp_y_score)
    cp_pr_auc = compute_pr_auc(cp_y_true, cp_y_score)
    wp_pr_base = compute_pr_baseline(wp_y_true)
    cp_pr_base = compute_pr_baseline(cp_y_true)

    return {
        "lower": lower,
        "upper": upper,
        "keep_ratio": keep_ratio,
        "min_features": min_features,
        "precision_floor": precision_floor,
        "beta": beta,
        **{f"lgbm_{k}": v for k, v in lgbm_params.items()},
        "full_feature_count": float(len(existing_features)),
        "selected_feature_count": float(len(selected_features)),
        "wp_f1_opt": wp_opt["f1"],
        "cp_f1_opt": cp_opt["f1"],
        "wp_fbeta_opt": wp_opt["f_beta"],
        "cp_fbeta_opt": cp_opt["f_beta"],
        "wp_threshold_opt": wp_opt["best_threshold"],
        "cp_threshold_opt": cp_opt["best_threshold"],
        "wp_precision_opt": wp_opt["precision"],
        "cp_precision_opt": cp_opt["precision"],
        "wp_recall_opt": wp_opt["recall"],
        "cp_recall_opt": cp_opt["recall"],
        "wp_pr_auc": wp_pr_auc,
        "cp_pr_auc": cp_pr_auc,
        "wp_pr_base": wp_pr_base,
        "cp_pr_base": cp_pr_base,
        "wp_pr_lift": (wp_pr_auc / wp_pr_base) if wp_pr_base > 0 else float("nan"),
        "cp_pr_lift": (cp_pr_auc / cp_pr_base) if cp_pr_base > 0 else float("nan"),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    print(f"Loaded data from '{DATA_PATH}' with shape {df.shape}")
    print(f"Output dir: {run_dir}")

    lower_upper_grid: List[Tuple[float, float]] = [
        (0.35, 0.65),
    ]
    # keep_ratio fixed to 0 by request; feature count is controlled only by min_features.
    keep_ratio_grid = [0.0]
    min_features_grid = [10, 12, 15, 20, 25, 30, 35]
    precision_floor_grid = [0.58, 0.60, 0.62, 0.65]
    beta_grid = [1.0, 0.8]
    lgbm_candidates = sample_lgbm_param_candidates(n_candidates=80, seed=RANDOM_STATE)
    max_trials = 4000

    rows: List[Dict[str, float]] = []
    total_possible = (
        len(lower_upper_grid)
        * len(keep_ratio_grid)
        * len(min_features_grid)
        * len(precision_floor_grid)
        * len(beta_grid)
        * len(lgbm_candidates)
    )
    total = min(total_possible, max_trials)
    all_combos = list(
        itertools.product(
            lower_upper_grid,
            keep_ratio_grid,
            min_features_grid,
            precision_floor_grid,
            beta_grid,
            lgbm_candidates,
        )
    )
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(all_combos)
    combos = all_combos[:max_trials]
    trial_id = 0
    for (lower, upper), keep_ratio, min_features, precision_floor, beta, lgbm_params in combos:
        trial_id += 1
        print(
            f"[{trial_id}/{total}] lower={lower:.2f}, upper={upper:.2f}, "
            f"keep_ratio={keep_ratio:.2f}, min_features={min_features}, "
            f"precision_floor={precision_floor:.2f}, beta={beta:.2f}, "
            f"num_leaves={lgbm_params['num_leaves']}, min_child_samples={lgbm_params['min_child_samples']}"
        )
        result = run_once(
            df,
            lower,
            upper,
            keep_ratio,
            min_features,
            precision_floor,
            beta,
            lgbm_params,
        )
        if not result:
            print("  -> skipped (insufficient data/folds)")
            continue
        rows.append(result)
        print(
            f"  -> cp_f1_opt={result['cp_f1_opt']:.3f}, cp_prec={result['cp_precision_opt']:.3f}, "
            f"cp_pr_lift={result['cp_pr_lift']:.3f}, wp_f1_opt={result['wp_f1_opt']:.3f}"
        )

    if not rows:
        print("No valid trials.")
        return

    df_res = pd.DataFrame(rows).sort_values(
        ["cp_f1_opt", "cp_pr_lift", "cp_precision_opt", "wp_f1_opt"],
        ascending=[False, False, False, False],
    )
    results_csv = run_dir / "trial_results.csv"
    df_res.to_csv(results_csv, index=False)

    best = df_res.iloc[0].to_dict()
    criteria_ok = (
        best["cp_f1_opt"] >= 0.75
        and best["wp_f1_opt"] >= 0.75
        and best["cp_pr_lift"] >= 1.30
        and best["wp_pr_lift"] >= 1.30
    )
    summary = {
        "success": bool(criteria_ok),
        "target": {
            "f1_min": 0.75,
            "pr_auc_lift_min": 1.30,
            "scope": "within_and_cross",
        },
        "best_trial": best,
        "results_csv": str(results_csv),
    }
    summary_json = run_dir / "best_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest trial:")
    print(json.dumps(best, indent=2))
    print(f"\nSaved trial table: {results_csv}")
    print(f"Saved summary: {summary_json}")
    print(f"Target reached: {criteria_ok}")


if __name__ == "__main__":
    main()

