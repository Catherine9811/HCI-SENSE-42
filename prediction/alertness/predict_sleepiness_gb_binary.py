from __future__ import annotations

"""
Binary sleepiness prediction script using Gradient Boosting,
based on per-participant percentile thresholds.

This is analogous to predict_sleepiness_forest_binary.py but
uses GradientBoostingClassifier instead of RandomForestClassifier.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

from prediction.alertness.predict_sleepiness_forest import (
    DATA_PATH,
    FEATURE_GROUPS,
    GROUP_COL,
    RANDOM_STATE,
)

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_TARGET_COL = "sleepiness"
BINARY_TARGET_COL = "sleepiness_binary"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ORIGINAL_TARGET_COL not in df.columns:
        raise ValueError(f"Original target column '{ORIGINAL_TARGET_COL}' not found in data.")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found in data.")
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


def build_gb_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build a preprocessing + GradientBoostingClassifier pipeline.
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

    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
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


def within_participant_cv_binary_gb(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    n_splits_default: int = 5,
) -> Dict[str, float]:
    X_all, y_all, groups_all = prepare_subset_with_target(df, feature_cols, target_col, group_col)

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_all[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    total_tp = total_fp = total_fn = total_tn = 0

    groups_dict = groups_all.groupby(groups_all).groups

    for _, idx in tqdm(
        groups_dict.items(),
        total=len(groups_dict),
        desc="  Within-participant CV (GB, binary): participants",
        leave=False,
    ):
        idx = np.asarray(list(idx))
        if idx.size < 3:
            continue
        n_splits = min(n_splits_default, idx.size)
        if n_splits < 2:
            continue

        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        X_p = X_all.iloc[idx]
        y_p = y_all.iloc[idx].to_numpy()

        for train_idx, test_idx in cv.split(X_p, y_p):
            y_train = y_p[train_idx]
            if len(np.unique(y_train)) < 2:
                # Skip folds where only one class is present in training data
                continue

            model = build_gb_pipeline(numeric_features, categorical_features)
            X_train, X_test = X_p.iloc[train_idx], X_p.iloc[test_idx]
            y_test = y_p[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            tp, fp, fn, tn, _, _, _ = compute_confusion_and_scores(y_test, y_pred)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    total = total_tp + total_fp + total_fn + total_tn
    if total == 0:
        return {}

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0.0 else 2 * precision * recall / (precision + recall)

    return {
        "TP": float(total_tp),
        "FP": float(total_fp),
        "FN": float(total_fn),
        "TN": float(total_tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def cross_participant_cv_binary_gb(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
) -> Dict[str, float]:
    X, y, groups = prepare_subset_with_target(df, feature_cols, target_col, group_col)

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    logo = LeaveOneGroupOut()

    total_tp = total_fp = total_fn = total_tn = 0

    progress = tqdm(
        total=groups.nunique(),
        desc="  Cross-participant CV (GB, binary): participants",
        leave=False,
    )

    for train_idx, test_idx in logo.split(X, y, groups):
        y_train = y.iloc[train_idx].to_numpy()
        if len(np.unique(y_train)) < 2:
            # Skip folds where only one class is present in training data
            progress.update(1)
            continue

        model = build_gb_pipeline(numeric_features, categorical_features)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        tp, fp, fn, tn, _, _, _ = compute_confusion_and_scores(y_test, y_pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        progress.update(1)

    progress.close()

    total = total_tp + total_fp + total_fn + total_tn
    if total == 0:
        return {}

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0.0 else 2 * precision * recall / (precision + recall)

    return {
        "TP": float(total_tp),
        "FP": float(total_fp),
        "FN": float(total_fn),
        "TN": float(total_tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_binary_gb_experiments(
    lower_percentile: float = 1.0 / 3.0,
    upper_percentile: float = 2.0 / 3.0,
) -> None:
    """
    Run binary classification experiments for all feature groups using Gradient Boosting
    and the given lower/upper percentiles.

    Example: lower=1/3, upper=2/3 → keep bottom/top 33% of samples per participant
    (low vs high sleepiness) and discard the middle 33%.
    """
    df = load_data(DATA_PATH)
    print(f"Loaded data from '{DATA_PATH}' with shape {df.shape}")
    print(
        f"Creating binary target '{BINARY_TARGET_COL}' from '{ORIGINAL_TARGET_COL}' "
        f"using within-participant lower/upper percentiles = "
        f"{lower_percentile:.2f}/{upper_percentile:.2f} "
        f"(keeping extremes, discarding middle)."
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
        desc="Feature groups (GB, binary)",
    ):
        if group_name != "mouse_keyboard_traits_sleep_engagement":
            continue
        existing_features = [f for f in raw_features if f in df_bin.columns]
        missing_features = sorted(set(raw_features) - set(existing_features))

        print("\n" + "=" * 80)
        print(f"Feature group (GB, binary): {group_name}")
        print(f"  Requested features: {len(raw_features)}")
        print(f"  Found in data:      {len(existing_features)}")
        print(f"  Missing in data:    {len(missing_features)}")
        if missing_features:
            print("  (Missing feature names will be ignored.)")

        if not existing_features:
            print("  No existing features for this group, skipping.")
            continue

        wp_results = within_participant_cv_binary_gb(
            df_bin,
            existing_features,
            target_col=BINARY_TARGET_COL,
            group_col=GROUP_COL,
        )
        cp_results = cross_participant_cv_binary_gb(
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

        print_results("Within-participant CV (GB, binary)", wp_results)
        print_results("Cross-participant CV (Leave-One-Participant-Out, GB, binary)", cp_results)


def main() -> None:
    # Default to keeping bottom/top 33% and discarding middle 33% per participant.
    run_binary_gb_experiments(lower_percentile=1.49 / 3.0, upper_percentile=1.51 / 3.0)


if __name__ == "__main__":
    main()

