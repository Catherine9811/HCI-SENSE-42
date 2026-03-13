from __future__ import annotations

"""
Binary sleepiness prediction script using Gradient Boosting,
based on absolute outcome thresholds.

Default absolute labeling:
  - sleepiness >= 5 -> positive class (1)
  - sleepiness <= 4 -> negative class (0)
  - values strictly between thresholds are dropped
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
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

ORIGINAL_TARGET_COL = "sleepiness"
BINARY_TARGET_COL = "sleepiness_binary"
POSITIVE_THRESHOLD = 8
NEGATIVE_THRESHOLD = 7

PR_CURVE_WITHIN_PATH = (
    BASE_DIR / "processed_data" / f"sleepiness_gb_binary_absolute_ge{POSITIVE_THRESHOLD}_le{NEGATIVE_THRESHOLD}_pr_curve_within.png"
)
PR_CURVE_CROSS_PATH = (
    BASE_DIR / "processed_data" / f"sleepiness_gb_binary_absolute_ge{POSITIVE_THRESHOLD}_le{NEGATIVE_THRESHOLD}_pr_curve_cross.png"
)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ORIGINAL_TARGET_COL not in df.columns:
        raise ValueError(f"Original target column '{ORIGINAL_TARGET_COL}' not found in data.")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found in data.")
    return df


def create_binary_target_absolute(
    df: pd.DataFrame,
    outcome_col: str,
    positive_threshold: float,
    negative_threshold: float,
    new_col: str = BINARY_TARGET_COL,
) -> pd.DataFrame:
    """
    Create a binary target from absolute thresholds:
      - outcome >= positive_threshold -> 1
      - outcome <= negative_threshold -> 0
      - otherwise NaN
    """
    if positive_threshold < negative_threshold:
        raise ValueError("positive_threshold must be >= negative_threshold.")
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in data.")

    df_bin = df.copy()
    outcome = df_bin[outcome_col]
    new_vals = np.full(df_bin.shape[0], np.nan)
    new_vals[outcome >= positive_threshold] = 1
    new_vals[outcome <= negative_threshold] = 0
    df_bin[new_col] = new_vals
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
    subset = subset.reset_index(drop=True)
    X = subset[feature_cols]
    y = subset[target_col]
    groups = subset[group_col]
    return X, y, groups


def build_gb_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
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

    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_depth=3,
        n_estimators=300,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_leaf=10,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def compute_confusion_and_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int, float, float, float]:
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

    total_tp = total_fp = total_fn = total_tn = 0
    all_y_true: List[np.ndarray] = []
    all_y_score: List[np.ndarray] = []

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
                continue

            model = build_gb_pipeline(numeric_features, categorical_features)
            X_train, X_test = X_p.iloc[train_idx], X_p.iloc[test_idx]
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

    logo = LeaveOneGroupOut()

    total_tp = total_fp = total_fn = total_tn = 0
    all_y_true: List[np.ndarray] = []
    all_y_score: List[np.ndarray] = []

    progress = tqdm(
        total=groups.nunique(),
        desc="  Cross-participant CV (GB, binary): participants",
        leave=False,
    )

    for train_idx, test_idx in logo.split(X, y, groups):
        y_train = y.iloc[train_idx].to_numpy()
        if len(np.unique(y_train)) < 2:
            progress.update(1)
            continue

        model = build_gb_pipeline(numeric_features, categorical_features)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()

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


def run_binary_gb_experiments_absolute(
    positive_threshold: float = POSITIVE_THRESHOLD,
    negative_threshold: float = NEGATIVE_THRESHOLD,
) -> None:
    """
    Run binary classification experiments for all feature groups using Gradient Boosting
    and absolute sleepiness thresholds.
    """
    df = load_data(DATA_PATH)
    print(f"Loaded data from '{DATA_PATH}' with shape {df.shape}")
    print(
        f"Creating binary target '{BINARY_TARGET_COL}' from '{ORIGINAL_TARGET_COL}' "
        f"using absolute thresholds: >= {positive_threshold:.2f} -> 1, "
        f"<= {negative_threshold:.2f} -> 0."
    )

    df_bin = create_binary_target_absolute(
        df,
        outcome_col=ORIGINAL_TARGET_COL,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        new_col=BINARY_TARGET_COL,
    )

    for group_name, raw_features in tqdm(
        FEATURE_GROUPS.items(),
        total=len(FEATURE_GROUPS),
        desc="Feature groups (GB, binary, absolute)",
    ):
        if group_name != "mouse_keyboard_traits_sleep_engagement":
            continue
        existing_features = [f for f in raw_features if f in df_bin.columns]
        missing_features = sorted(set(raw_features) - set(existing_features))

        print("\n" + "=" * 80)
        print(f"Feature group (GB, binary, absolute): {group_name}")
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

        print_results("Within-participant CV (GB, binary, absolute)", wp_results)
        print_results("Cross-participant CV (Leave-One-Participant-Out, GB, binary, absolute)", cp_results)

        if wp_y_true.size > 0 and wp_y_score.size > 0:
            wp_pr_auc = save_pr_curve(
                wp_y_true,
                wp_y_score,
                "Within-participant PR Curve (GB, binary, absolute)",
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
                "Cross-participant PR Curve (GB, binary, absolute)",
                PR_CURVE_CROSS_PATH,
            )
            print(f"  Cross-participant PR-AUC={cp_pr_auc:.3f}")
            print(f"  Saved PR curve: {PR_CURVE_CROSS_PATH}")
        else:
            print("  Cross-participant PR curve not saved (no valid probability outputs).")


def main() -> None:
    run_binary_gb_experiments_absolute(
        positive_threshold=POSITIVE_THRESHOLD,
        negative_threshold=NEGATIVE_THRESHOLD,
    )


if __name__ == "__main__":
    main()

