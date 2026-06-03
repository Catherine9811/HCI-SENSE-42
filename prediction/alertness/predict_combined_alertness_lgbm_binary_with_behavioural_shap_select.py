from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
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
from prediction.alertness.shared_config import DATA_PATH, FEATURE_GROUPS


BASE_DIR = Path(__file__).resolve().parent
TARGET_GROUP = "mouse_keyboard_traits_sleep_engagement_behavioural"
SHAP_IMPORTANCE_PATH = (
    BASE_DIR / "processed_data" / "combined_alertness_lgbm_with_behavioural_shap_importance.csv"
)
SELECTED_FEATURES_PATH = (
    BASE_DIR / "processed_data" / "combined_alertness_lgbm_with_behavioural_selected_features.csv"
)
LGBM_FIXED_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 2,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
}


def _get_transformed_feature_names(
    preprocessor,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    names = list(numeric_features)
    if categorical_features:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        if hasattr(onehot, "get_feature_names_out"):
            cat_names = onehot.get_feature_names_out(categorical_features)
        else:
            cat_names = onehot.get_feature_names(categorical_features)
        names.extend(list(cat_names))
    return names


def _map_transformed_to_raw_feature(
    transformed_name: str,
    numeric_features: List[str],
    categorical_features: List[str],
) -> str:
    if transformed_name in numeric_features:
        return transformed_name
    for col in categorical_features:
        if transformed_name.startswith(f"{col}_"):
            return col
    return transformed_name


def compute_shap_raw_feature_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    sample_size: int = 1200,
) -> pd.DataFrame:
    X, y, groups = prepare_subset_with_target(df, feature_cols, target_col, group_col)
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    scaler_stats = fit_participant_iqr_scaler(X, groups, numeric_features)
    X_scaled = transform_with_participant_iqr_scaler(X, groups, numeric_features, *scaler_stats)

    pipeline = build_lgbm_pipeline_with_params(
        numeric_features,
        categorical_features,
        LGBM_FIXED_PARAMS,
    )
    pipeline.fit(X_scaled, y.to_numpy())

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    rng = np.random.RandomState(RANDOM_STATE)
    if len(X_scaled) > sample_size:
        sample_idx = rng.choice(len(X_scaled), size=sample_size, replace=False)
        X_sample = X_scaled.iloc[sample_idx]
    else:
        X_sample = X_scaled

    X_trans = preprocessor.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    transformed_feature_names = _get_transformed_feature_names(
        preprocessor, numeric_features, categorical_features
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    transformed_imp = pd.DataFrame(
        {
            "transformed_feature": transformed_feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    )
    transformed_imp["raw_feature"] = transformed_imp["transformed_feature"].apply(
        lambda n: _map_transformed_to_raw_feature(n, numeric_features, categorical_features)
    )
    raw_imp = (
        transformed_imp.groupby("raw_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return raw_imp


def select_features_by_shap(
    importance_df: pd.DataFrame,
    keep_ratio: float = 0.5,
    min_features: int = 15,
) -> List[str]:
    if importance_df.empty:
        return []
    n_total = len(importance_df)
    n_keep = max(min_features, ceil(n_total * keep_ratio))
    n_keep = min(n_keep, n_total)
    return importance_df["raw_feature"].head(n_keep).tolist()


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


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    precision_floor: float = 0.0,
    beta: float = 1.0,
) -> Dict[str, float]:
    if y_true.size == 0 or y_score.size == 0:
        return {
            "best_threshold": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    beta2 = beta * beta
    best_feasible: Optional[Dict[str, float]] = None
    best_any: Optional[Dict[str, float]] = None
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_score >= t).astype(int)
        _, _, _, _, precision, recall, f1 = _compute_confusion_and_scores(y_true, y_pred)
        denom = beta2 * precision + recall
        f_beta = 0.0 if denom == 0.0 else (1.0 + beta2) * precision * recall / denom
        row = {
            "best_threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f_beta": float(f_beta),
        }
        if best_any is None or row["f_beta"] > best_any["f_beta"]:
            best_any = row
        if precision >= precision_floor:
            if best_feasible is None or row["f_beta"] > best_feasible["f_beta"]:
                best_feasible = row
    return best_feasible if best_feasible is not None else best_any


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
        X_all,
        groups_all,
        numeric_features,
        *scaler_stats,
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

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        for train_idx, test_idx in cv.split(X_p, y_p):
            y_train = y_p[train_idx]
            if len(np.unique(y_train)) < 2:
                continue
            model = build_lgbm_pipeline_with_params(
                numeric_features,
                categorical_features,
                lgbm_params,
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
    X_scaled = transform_with_participant_iqr_scaler(
        X,
        groups,
        numeric_features,
        *scaler_stats,
    )
    logo = LeaveOneGroupOut()

    total_tp = total_fp = total_fn = total_tn = 0
    all_y_true: List[np.ndarray] = []
    all_y_score: List[np.ndarray] = []
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        y_train = y.iloc[train_idx].to_numpy().astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        model = build_lgbm_pipeline_with_params(
            numeric_features,
            categorical_features,
            lgbm_params,
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


def _print_result_block(title: str, results: Dict[str, float]) -> None:
    print(f"  {title}:")
    if not results:
        print("    (no valid folds)")
        return
    print(
        f"    TP={results['TP']:.0f}, FP={results['FP']:.0f}, "
        f"FN={results['FN']:.0f}, TN={results['TN']:.0f}"
    )
    print(
        f"    precision={results['precision']:.3f}, "
        f"recall={results['recall']:.3f}, "
        f"f1={results['f1']:.3f}"
    )


def _compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or y_score.size == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def _compute_pr_baseline(y_true: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true))


def run_shap_feature_selection_experiment(
    lower_percentile: float = 0.35,
    upper_percentile: float = 0.65,
    keep_ratio: float = 0.0,
    min_features: int = 15,
    precision_floor: float = 0.60,
    beta: float = 1.00,
) -> None:
    df = load_data(DATA_PATH)
    print(f"Loaded data from '{DATA_PATH}' with shape {df.shape}")
    df_bin = create_alertness_binary_target_per_participant(
        df=df,
        group_col=GROUP_COL,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        new_col=BINARY_TARGET_COL,
    )

    raw_features = FEATURE_GROUPS.get(TARGET_GROUP, [])
    existing_features = [f for f in raw_features if f in df_bin.columns]
    if not existing_features:
        print(f"No existing features found for group '{TARGET_GROUP}'.")
        return

    print("\n" + "=" * 80)
    print(f"Feature group: {TARGET_GROUP}")
    print(f"  Full feature count: {len(existing_features)}")

    baseline_wp, baseline_wp_y_true, baseline_wp_y_score = within_participant_cv_binary_lgbm_with_params(
        df_bin,
        existing_features,
        target_col=BINARY_TARGET_COL,
        group_col=GROUP_COL,
        lgbm_params=LGBM_FIXED_PARAMS,
    )
    baseline_cp, baseline_cp_y_true, baseline_cp_y_score = cross_participant_cv_binary_lgbm_with_params(
        df_bin,
        existing_features,
        target_col=BINARY_TARGET_COL,
        group_col=GROUP_COL,
        lgbm_params=LGBM_FIXED_PARAMS,
    )

    shap_importance = compute_shap_raw_feature_importance(
        df=df_bin,
        feature_cols=existing_features,
        target_col=BINARY_TARGET_COL,
        group_col=GROUP_COL,
    )
    SHAP_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    shap_importance.to_csv(SHAP_IMPORTANCE_PATH, index=False)

    selected_features = select_features_by_shap(
        shap_importance, keep_ratio=keep_ratio, min_features=min_features
    )
    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        SELECTED_FEATURES_PATH, index=False
    )
    print("  Selected features:")
    for i, feature_name in enumerate(selected_features, start=1):
        print(f"    {i:2d}. {feature_name}")

    reduced_wp, reduced_wp_y_true, reduced_wp_y_score = within_participant_cv_binary_lgbm_with_params(
        df_bin,
        selected_features,
        target_col=BINARY_TARGET_COL,
        group_col=GROUP_COL,
        lgbm_params=LGBM_FIXED_PARAMS,
    )
    reduced_cp, reduced_cp_y_true, reduced_cp_y_score = cross_participant_cv_binary_lgbm_with_params(
        df_bin,
        selected_features,
        target_col=BINARY_TARGET_COL,
        group_col=GROUP_COL,
        lgbm_params=LGBM_FIXED_PARAMS,
    )

    baseline_wp_pr_auc = _compute_pr_auc(baseline_wp_y_true, baseline_wp_y_score)
    baseline_cp_pr_auc = _compute_pr_auc(baseline_cp_y_true, baseline_cp_y_score)
    reduced_wp_pr_auc = _compute_pr_auc(reduced_wp_y_true, reduced_wp_y_score)
    reduced_cp_pr_auc = _compute_pr_auc(reduced_cp_y_true, reduced_cp_y_score)
    baseline_wp_pr_base = _compute_pr_baseline(baseline_wp_y_true)
    baseline_cp_pr_base = _compute_pr_baseline(baseline_cp_y_true)
    reduced_wp_pr_base = _compute_pr_baseline(reduced_wp_y_true)
    reduced_cp_pr_base = _compute_pr_baseline(reduced_cp_y_true)
    baseline_wp_opt = threshold_sweep(
        baseline_wp_y_true, baseline_wp_y_score, precision_floor=precision_floor, beta=beta
    )
    baseline_cp_opt = threshold_sweep(
        baseline_cp_y_true, baseline_cp_y_score, precision_floor=precision_floor, beta=beta
    )
    reduced_wp_opt = threshold_sweep(
        reduced_wp_y_true, reduced_wp_y_score, precision_floor=precision_floor, beta=beta
    )
    reduced_cp_opt = threshold_sweep(
        reduced_cp_y_true, reduced_cp_y_score, precision_floor=precision_floor, beta=beta
    )

    removed_features = [f for f in existing_features if f not in set(selected_features)]
    print(f"  Selected features after SHAP filtering: {len(selected_features)}")
    print(f"  Removed features: {len(removed_features)}")
    if removed_features:
        print("  Example removed features:")
        for name in removed_features[:20]:
            print(f"    - {name}")
        if len(removed_features) > 20:
            print(f"    ... and {len(removed_features) - 20} more")

    behavioural_in_selected = [
        f for f in selected_features if f in FEATURE_GROUPS.get("behavioural", [])
    ]
    print(f"  Behavioural features retained after SHAP: {len(behavioural_in_selected)}")
    if behavioural_in_selected:
        for f in behavioural_in_selected:
            row = shap_importance[shap_importance["raw_feature"] == f]
            imp = row["mean_abs_shap"].values[0] if not row.empty else float("nan")
            rank = int(shap_importance[shap_importance["raw_feature"] == f].index[0]) + 1 if not row.empty else "?"
            print(f"    - {f}  (rank {rank}, SHAP={imp:.4f})")

    print("\nResults BEFORE feature removal:")
    _print_result_block("Within-participant CV", baseline_wp)
    _print_result_block("Cross-participant CV (Leave-One-Participant-Out)", baseline_cp)
    print(f"  PR-AUC (within): {baseline_wp_pr_auc:.3f}")
    print(f"  PR-AUC (cross):  {baseline_cp_pr_auc:.3f}")
    print(f"  PR baseline (within): {baseline_wp_pr_base:.3f}")
    print(f"  PR baseline (cross):  {baseline_cp_pr_base:.3f}")
    print(
        f"  Threshold-opt (within) @ precision_floor={precision_floor:.2f}, beta={beta:.2f}: "
        f"t={baseline_wp_opt['best_threshold']:.2f}, f1={baseline_wp_opt['f1']:.3f}, "
        f"precision={baseline_wp_opt['precision']:.3f}, recall={baseline_wp_opt['recall']:.3f}"
    )
    print(
        f"  Threshold-opt (cross)  @ precision_floor={precision_floor:.2f}, beta={beta:.2f}: "
        f"t={baseline_cp_opt['best_threshold']:.2f}, f1={baseline_cp_opt['f1']:.3f}, "
        f"precision={baseline_cp_opt['precision']:.3f}, recall={baseline_cp_opt['recall']:.3f}"
    )

    print("\nResults AFTER feature removal:")
    _print_result_block("Within-participant CV", reduced_wp)
    _print_result_block("Cross-participant CV (Leave-One-Participant-Out)", reduced_cp)
    print(f"  PR-AUC (within): {reduced_wp_pr_auc:.3f}")
    print(f"  PR-AUC (cross):  {reduced_cp_pr_auc:.3f}")
    print(f"  PR baseline (within): {reduced_wp_pr_base:.3f}")
    print(f"  PR baseline (cross):  {reduced_cp_pr_base:.3f}")
    print(
        f"  Threshold-opt (within) @ precision_floor={precision_floor:.2f}, beta={beta:.2f}: "
        f"t={reduced_wp_opt['best_threshold']:.2f}, f1={reduced_wp_opt['f1']:.3f}, "
        f"precision={reduced_wp_opt['precision']:.3f}, recall={reduced_wp_opt['recall']:.3f}"
    )
    print(
        f"  Threshold-opt (cross)  @ precision_floor={precision_floor:.2f}, beta={beta:.2f}: "
        f"t={reduced_cp_opt['best_threshold']:.2f}, f1={reduced_cp_opt['f1']:.3f}, "
        f"precision={reduced_cp_opt['precision']:.3f}, recall={reduced_cp_opt['recall']:.3f}"
    )

    print(f"\nSaved SHAP raw-feature importance: {SHAP_IMPORTANCE_PATH}")
    print(f"Saved selected feature list: {SELECTED_FEATURES_PATH}")


def main() -> None:
    run_shap_feature_selection_experiment(
        # [1910/4000] lower=0.35, upper=0.65, keep_ratio=0.00, min_features=15,
        # precision_floor=0.60, beta=1.00, num_leaves=7, min_child_samples=2
        lower_percentile=0.35,
        upper_percentile=0.65,
        keep_ratio=0.0,
        min_features=15,
        precision_floor=0.60,
        beta=1.00,
    )


if __name__ == "__main__":
    main()
