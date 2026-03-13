from __future__ import annotations

"""
Binary TLX prediction script using Gradient Boosting,
based on per-participant percentile thresholds and participant-level
standardized numeric features.

TLX (Task Load Index) is calculated as:
  tlx = temporal_demand + mental_demand + effort + frustration - performance

This script uses LightGBM (LGBMClassifier).
Numeric features are rescaled with participant-level median/IQR, and
IQR is stabilized with training-fold global IQR fallback to avoid
division by near-zero values.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

from prediction.alertness.shared_config import (
    DATA_PATH,
    FEATURE_GROUPS,
    GROUP_COL,
    RANDOM_STATE,
)

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_TARGET_COL = "tlx"
BINARY_TARGET_COL = "tlx_binary"
PR_CURVE_WITHIN_PATH = (
    BASE_DIR / "processed_data" / "tlx_gb_binary_participant_iqr_scaled_pr_curve_within.png"
)
PR_CURVE_CROSS_PATH = (
    BASE_DIR / "processed_data" / "tlx_gb_binary_participant_iqr_scaled_pr_curve_cross.png"
)


def load_data(path: Path) -> pd.DataFrame:
    """
    Load data and calculate TLX (Task Load Index).
    """
    df = pd.read_csv(path)

    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found in data.")

    required_cols = [
        "temporal_demand",
        "mental_demand",
        "effort",
        "frustration",
        "performance",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns for TLX calculation not found in data: {missing_cols}"
        )

    df[ORIGINAL_TARGET_COL] = (
        df["temporal_demand"]
        + df["mental_demand"]
        + df["effort"]
        + df["frustration"]
        - df["performance"]
    )
    return df


def create_binary_target_per_participant(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    lower_percentile: float,
    upper_percentile: float,
    new_col: str = BINARY_TARGET_COL,
) -> pd.DataFrame:
    """
    Create a binary target for each participant based on within-participant percentiles.

    For each participant:
      - Compute the lower and upper percentiles of the outcome variable (ignoring NaNs)
      - Label samples with outcome <= lower_threshold as 0
      - Label samples with outcome >= upper_threshold as 1
      - Samples in between are set to NaN and will be dropped later
    """
    if not (0.0 < lower_percentile < upper_percentile < 1.0):
        raise ValueError(
            "lower_percentile and upper_percentile must satisfy 0 < lower < upper < 1, "
            "e.g., 0.33 and 0.67 to keep bottom/top 33% and discard middle 33%."
        )

    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in data.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in data.")

    df_bin = df.copy()

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        vals = g[outcome_col].dropna().to_numpy()
        if vals.size == 0:
            g[new_col] = np.nan
            return g
        lower_th = np.quantile(vals, lower_percentile)
        upper_th = np.quantile(vals, upper_percentile)

        # Start with NaNs for all rows, then fill 0/1 for extremes
        new_vals = np.full(g.shape[0], np.nan)
        new_vals[g[outcome_col] <= lower_th] = 0
        new_vals[g[outcome_col] >= upper_th] = 1

        g[new_col] = new_vals
        return g

    # Use group_keys=False so that the original index/columns are preserved;
    # include_groups is not available in older pandas versions.
    df_bin = df_bin.groupby(group_col, group_keys=False).apply(_per_group)
    return df_bin


def prepare_subset_with_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    cols = list(dict.fromkeys(feature_cols + [target_col, group_col]))
    subset = df[cols].copy()
    subset = subset.dropna(subset=[target_col, group_col])
    # Reset index so that subsequent use of iloc with group indices is safe
    subset = subset.reset_index(drop=True)
    X = subset[feature_cols]
    y = subset[target_col]
    groups = subset[group_col]
    return X, y, groups


def fit_participant_iqr_scaler(
    X_train: pd.DataFrame,
    groups_train: pd.Series,
    numeric_features: List[str],
    min_iqr: float = 1e-6,
    iqr_floor_fraction: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Fit participant-level median/IQR scaler on a reference dataset.

    Stable IQR is enforced by a floor:
      max(global_iqr * iqr_floor_fraction, min_iqr)
    """
    if not numeric_features:
        empty_df = pd.DataFrame(index=groups_train.unique())
        empty_series = pd.Series(dtype=float)
        return empty_df, empty_df, empty_series, empty_series

    x_num = X_train[numeric_features]
    participant_medians = x_num.groupby(groups_train).median()
    participant_q75 = x_num.groupby(groups_train).quantile(0.75)
    participant_q25 = x_num.groupby(groups_train).quantile(0.25)
    participant_iqrs = participant_q75 - participant_q25

    global_median = x_num.median(axis=0, skipna=True)
    global_iqr = x_num.quantile(0.75, axis=0) - x_num.quantile(0.25, axis=0)
    global_iqr = global_iqr.replace([np.inf, -np.inf], np.nan)

    stable_iqr_floor = np.maximum(global_iqr.to_numpy(dtype=float) * iqr_floor_fraction, min_iqr)
    stable_iqr_floor = pd.Series(stable_iqr_floor, index=numeric_features)

    global_iqr_stable = global_iqr.where(global_iqr > stable_iqr_floor, other=stable_iqr_floor)
    global_iqr_stable = global_iqr_stable.fillna(stable_iqr_floor).astype(float)

    participant_iqrs = participant_iqrs.replace([np.inf, -np.inf], np.nan)
    participant_iqrs = participant_iqrs.where(participant_iqrs > stable_iqr_floor, other=np.nan)
    participant_iqrs = participant_iqrs.fillna(global_iqr_stable)

    global_median = global_median.fillna(0.0).astype(float)
    return participant_medians, participant_iqrs, global_median, global_iqr_stable


def transform_with_participant_iqr_scaler(
    X: pd.DataFrame,
    groups: pd.Series,
    numeric_features: List[str],
    participant_medians: pd.DataFrame,
    participant_iqrs: pd.DataFrame,
    global_median: pd.Series,
    global_iqr_stable: pd.Series,
) -> pd.DataFrame:
    """
    Apply participant-level robust scaling to numeric features:
    (x - participant_median) / stable_participant_iqr.
    """
    if not numeric_features:
        return X

    X_scaled = X.copy()
    x_num = X_scaled[numeric_features].copy()

    row_medians = participant_medians.reindex(groups).set_index(X_scaled.index)
    row_iqrs = participant_iqrs.reindex(groups).set_index(X_scaled.index)

    global_median_df = pd.DataFrame(
        np.tile(global_median.to_numpy(), (len(X_scaled), 1)),
        columns=numeric_features,
        index=X_scaled.index,
    )
    global_iqr_df = pd.DataFrame(
        np.tile(global_iqr_stable.to_numpy(), (len(X_scaled), 1)),
        columns=numeric_features,
        index=X_scaled.index,
    )

    row_medians = row_medians.fillna(global_median_df)
    row_iqrs = row_iqrs.fillna(global_iqr_df)

    X_scaled.loc[:, numeric_features] = (x_num - row_medians) / row_iqrs
    return X_scaled


def build_lgbm_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build a preprocessing + LightGBM classifier pipeline.
    """
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

    model = LGBMClassifier(
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=5,
        min_split_gain=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=-1,
        verbosity=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return clf


def compute_confusion_and_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int, float, float, float]:
    """
    Return TP, FP, FN, TN, precision, recall, f1 for binary labels {0,1}.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = float(precision_score(y_true, y_pred, zero_division=0))

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = float(recall_score(y_true, y_pred, zero_division=0))

    if (precision + recall) == 0.0:
        f1 = 0.0
    else:
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return tp, fp, fn, tn, precision, recall, f1


def save_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    output_path: Path,
) -> float:
    if y_true.size == 0 or y_score.size == 0:
        return float("nan")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recall, precision))
    baseline_precision = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    ax.plot(recall, precision, color="#1f77b4", linewidth=2.0, label=f"PR-AUC = {pr_auc:.3f}")
    ax.hlines(
        y=baseline_precision,
        xmin=0.0,
        xmax=1.0,
        colors="#d62728",
        linestyles="--",
        linewidth=1.8,
        label=f"No-skill baseline = {baseline_precision:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="lower left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pr_auc


def within_participant_cv_binary_gb(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    n_splits_default: int = 5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    X_all, y_all, groups_all = prepare_subset_with_target(df, feature_cols, target_col, group_col)

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_all[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    # Fit once on all rows and transform the full dataset before any CV split.
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

    for _, idx in tqdm(
        groups_dict.items(),
        total=len(groups_dict),
        desc="  Within-participant CV (GB, binary, participant-IQR scaled): participants",
        leave=False,
    ):
        idx = np.asarray(list(idx))
        if idx.size < 3:
            continue
        n_splits = min(n_splits_default, idx.size)
        if n_splits < 2:
            continue

        X_p = X_all_scaled.iloc[idx]
        y_p = y_all.iloc[idx].to_numpy()
        class_counts = np.bincount(y_p.astype(int), minlength=2)
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
                # Skip folds where only one class is present in training data
                continue

            X_train, X_test = X_p.iloc[train_idx], X_p.iloc[test_idx]

            model = build_lgbm_pipeline(numeric_features, categorical_features)
            y_test = y_p[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]

            tp, fp, fn, tn, _, _, _ = compute_confusion_and_scores(y_test, y_pred)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            all_y_true.append(np.asarray(y_test).astype(int))
            all_y_score.append(np.asarray(y_score).astype(float))

    total = total_tp + total_fp + total_fn + total_tn
    if total == 0:
        return {}, np.array([], dtype=int), np.array([], dtype=float)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0.0 else 2 * precision * recall / (precision + recall)

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


def cross_participant_cv_binary_gb(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    X, y, groups = prepare_subset_with_target(df, feature_cols, target_col, group_col)

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    # Fit once on all rows and transform the full dataset before any CV split.
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

    progress = tqdm(
        total=groups.nunique(),
        desc="  Cross-participant CV (GB, binary, participant-IQR scaled): participants",
        leave=False,
    )

    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        y_train = y.iloc[train_idx].to_numpy()
        if len(np.unique(y_train)) < 2:
            # Skip folds where only one class is present in training data
            progress.update(1)
            continue

        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()

        model = build_lgbm_pipeline(numeric_features, categorical_features)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        tp, fp, fn, tn, _, _, _ = compute_confusion_and_scores(y_test, y_pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        all_y_true.append(np.asarray(y_test).astype(int))
        all_y_score.append(np.asarray(y_score).astype(float))

        progress.update(1)

    progress.close()

    total = total_tp + total_fp + total_fn + total_tn
    if total == 0:
        return {}, np.array([], dtype=int), np.array([], dtype=float)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0.0 else 2 * precision * recall / (precision + recall)

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


def run_binary_gb_experiments(
    lower_percentile: float = 1.0 / 3.0,
    upper_percentile: float = 2.0 / 3.0,
) -> None:
    """
    Run binary classification experiments for all feature groups using
    Gradient Boosting and participant-level IQR-scaled numeric features.

    Example: lower=1/3, upper=2/3 → keep bottom/top 33% of samples per participant
    (low vs high TLX) and discard the middle 33%.
    """
    df = load_data(DATA_PATH)
    print(f"Loaded data from '{DATA_PATH}' with shape {df.shape}")
    print(
        f"Creating binary target '{BINARY_TARGET_COL}' from '{ORIGINAL_TARGET_COL}' "
        f"using within-participant lower/upper percentiles = "
        f"{lower_percentile:.2f}/{upper_percentile:.2f} "
        f"(keeping extremes, discarding middle)."
    )
    print(
        "Applying participant-level median/IQR scaling on the full dataset first, "
        "then using the pre-scaled data directly for both validation schemes."
    )

    df_bin = create_binary_target_per_participant(
        df,
        outcome_col=ORIGINAL_TARGET_COL,
        group_col=GROUP_COL,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        new_col=BINARY_TARGET_COL,
    )

    for group_name, raw_features in tqdm(
        FEATURE_GROUPS.items(),
        total=len(FEATURE_GROUPS),
        desc="Feature groups (GB, binary, participant-IQR scaled)",
    ):
        if group_name != "mouse_keyboard_traits_sleep_engagement":
            continue
        existing_features = [f for f in raw_features if f in df_bin.columns]
        missing_features = sorted(set(raw_features) - set(existing_features))

        print("\n" + "=" * 80)
        print(f"Feature group (GB, binary, participant-IQR scaled): {group_name}")
        print(f"  Requested features: {len(raw_features)}")
        print(f"  Found in data:      {len(existing_features)}")
        print(f"  Missing in data:    {len(missing_features)}")
        if missing_features:
            print("  (Missing feature names will be ignored.)")

        if not existing_features:
            print("  No existing features for this group, skipping.")
            continue

        wp_results, wp_y_true, wp_y_score = within_participant_cv_binary_gb(
            df_bin,
            existing_features,
            target_col=BINARY_TARGET_COL,
            group_col=GROUP_COL,
        )
        cp_results, cp_y_true, cp_y_score = cross_participant_cv_binary_gb(
            df_bin,
            existing_features,
            target_col=BINARY_TARGET_COL,
            group_col=GROUP_COL,
        )

        if not wp_results and not cp_results:
            print("  Not enough data for the requested CV schemes.")
            continue

        def print_results(title: str, results: Dict[str, float]) -> None:
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

        print_results("Within-participant CV (GB, binary, participant-IQR scaled)", wp_results)
        print_results(
            "Cross-participant CV (Leave-One-Participant-Out, GB, participant-IQR scaled)",
            cp_results,
        )

        if wp_y_true.size > 0 and wp_y_score.size > 0:
            wp_pr_auc = save_pr_curve(
                wp_y_true,
                wp_y_score,
                "Within-participant PR Curve (GB, binary, participant-IQR scaled)",
                PR_CURVE_WITHIN_PATH,
            )
            print(f"  Within-participant PR-AUC={wp_pr_auc:.3f}")
            print(f"  Saved PR curve: {PR_CURVE_WITHIN_PATH}")
        else:
            print("  Within-participant PR curve not saved (no valid probability outputs).")

        if cp_y_true.size > 0 and cp_y_score.size > 0:
            cp_pr_auc = save_pr_curve(
                cp_y_true,
                cp_y_score,
                "Cross-participant PR Curve (GB, binary, participant-IQR scaled)",
                PR_CURVE_CROSS_PATH,
            )
            print(f"  Cross-participant PR-AUC={cp_pr_auc:.3f}")
            print(f"  Saved PR curve: {PR_CURVE_CROSS_PATH}")
        else:
            print("  Cross-participant PR curve not saved (no valid probability outputs).")


def main() -> None:
    # Default to keeping bottom/top 33% and discarding middle 33% per participant.
    run_binary_gb_experiments(lower_percentile=0.33, upper_percentile=0.66)


if __name__ == "__main__":
    main()

