from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import auc, precision_recall_curve

from prediction.alertness.predict_combined_alertness_lgbm_binary_participant_iqr_scaled import (
    BINARY_TARGET_COL,
    GROUP_COL,
    RANDOM_STATE,
    build_lgbm_pipeline,
    create_alertness_binary_target_per_participant,
    fit_participant_iqr_scaler,
    load_data,
    prepare_subset_with_target,
    transform_with_participant_iqr_scaler,
    within_participant_cv_binary_lgbm,
    cross_participant_cv_binary_lgbm,
)
from prediction.alertness.shared_config import DATA_PATH, FEATURE_GROUPS


BASE_DIR = Path(__file__).resolve().parent
TARGET_GROUP = "mouse_keyboard_traits_sleep_engagement"
SHAP_IMPORTANCE_PATH = (
    BASE_DIR / "processed_data" / "combined_alertness_lgbm_participant_iqr_scaled_shap_importance.csv"
)
SELECTED_FEATURES_PATH = (
    BASE_DIR / "processed_data" / "combined_alertness_lgbm_participant_iqr_scaled_selected_features.csv"
)


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

    pipeline = build_lgbm_pipeline(numeric_features, categorical_features)
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
    lower_percentile: float = 0.33,
    upper_percentile: float = 0.66,
    keep_ratio: float = 0.5,
    min_features: int = 15,
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

    baseline_wp, baseline_wp_y_true, baseline_wp_y_score = within_participant_cv_binary_lgbm(
        df_bin, existing_features, target_col=BINARY_TARGET_COL, group_col=GROUP_COL
    )
    baseline_cp, baseline_cp_y_true, baseline_cp_y_score = cross_participant_cv_binary_lgbm(
        df_bin, existing_features, target_col=BINARY_TARGET_COL, group_col=GROUP_COL
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

    # Refit/evaluate after feature removal using the reduced feature set.
    reduced_wp, reduced_wp_y_true, reduced_wp_y_score = within_participant_cv_binary_lgbm(
        df_bin, selected_features, target_col=BINARY_TARGET_COL, group_col=GROUP_COL
    )
    reduced_cp, reduced_cp_y_true, reduced_cp_y_score = cross_participant_cv_binary_lgbm(
        df_bin, selected_features, target_col=BINARY_TARGET_COL, group_col=GROUP_COL
    )

    baseline_wp_pr_auc = _compute_pr_auc(baseline_wp_y_true, baseline_wp_y_score)
    baseline_cp_pr_auc = _compute_pr_auc(baseline_cp_y_true, baseline_cp_y_score)
    reduced_wp_pr_auc = _compute_pr_auc(reduced_wp_y_true, reduced_wp_y_score)
    reduced_cp_pr_auc = _compute_pr_auc(reduced_cp_y_true, reduced_cp_y_score)
    baseline_wp_pr_base = _compute_pr_baseline(baseline_wp_y_true)
    baseline_cp_pr_base = _compute_pr_baseline(baseline_cp_y_true)
    reduced_wp_pr_base = _compute_pr_baseline(reduced_wp_y_true)
    reduced_cp_pr_base = _compute_pr_baseline(reduced_cp_y_true)

    removed_features = [f for f in existing_features if f not in set(selected_features)]
    print(f"  Selected features after SHAP filtering: {len(selected_features)}")
    print(f"  Removed features: {len(removed_features)}")
    if removed_features:
        print("  Example removed features:")
        for name in removed_features[:20]:
            print(f"    - {name}")
        if len(removed_features) > 20:
            print(f"    ... and {len(removed_features) - 20} more")

    print("\nResults BEFORE feature removal:")
    _print_result_block("Within-participant CV", baseline_wp)
    _print_result_block("Cross-participant CV (Leave-One-Participant-Out)", baseline_cp)
    print(f"  PR-AUC (within): {baseline_wp_pr_auc:.3f}")
    print(f"  PR-AUC (cross):  {baseline_cp_pr_auc:.3f}")
    print(f"  PR baseline (within): {baseline_wp_pr_base:.3f}")
    print(f"  PR baseline (cross):  {baseline_cp_pr_base:.3f}")

    print("\nResults AFTER feature removal:")
    _print_result_block("Within-participant CV", reduced_wp)
    _print_result_block("Cross-participant CV (Leave-One-Participant-Out)", reduced_cp)
    print(f"  PR-AUC (within): {reduced_wp_pr_auc:.3f}")
    print(f"  PR-AUC (cross):  {reduced_cp_pr_auc:.3f}")
    print(f"  PR baseline (within): {reduced_wp_pr_base:.3f}")
    print(f"  PR baseline (cross):  {reduced_cp_pr_base:.3f}")

    print(f"\nSaved SHAP raw-feature importance: {SHAP_IMPORTANCE_PATH}")
    print(f"Saved selected feature list: {SELECTED_FEATURES_PATH}")


def main() -> None:
    run_shap_feature_selection_experiment(
        lower_percentile=0.45,
        upper_percentile=0.55,
        keep_ratio=0.1,
        min_features=15,
    )


if __name__ == "__main__":
    main()

