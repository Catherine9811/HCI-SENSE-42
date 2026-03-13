from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

from prediction.alertness.shared_config import (
    DATA_PATH,
    FEATURE_GROUPS,
    GROUP_COL,
    RANDOM_STATE,
    TARGET_COL,
)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found in data.")
    return df


def infer_task_type(y: pd.Series) -> str:
    """Heuristic: small integer-valued target → classification, otherwise regression."""
    if pd.api.types.is_numeric_dtype(y):
        unique_vals = np.unique(y.dropna())
        if len(unique_vals) <= 6 and np.all(np.mod(unique_vals, 1) == 0):
            return "classification"
        return "regression"
    return "classification"


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    task_type: str,
    random_state: int = RANDOM_STATE,
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

    if task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
        )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return clf


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, task_type: str
) -> Dict[str, float]:
    if task_type == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        }
    else:
        return {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
        }


def summarize_metrics(
    metrics: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    summary: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics], dtype=float)
        summary[k] = (float(np.mean(vals)), float(np.std(vals)))
    return summary


def prepare_subset(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    cols = list(dict.fromkeys(feature_cols + [TARGET_COL, GROUP_COL]))
    subset = df[cols].copy()
    # Drop rows with missing target or group; features are handled by imputers
    subset = subset.dropna(subset=[TARGET_COL, GROUP_COL])
    X = subset[feature_cols]
    y = subset[TARGET_COL]
    groups = subset[GROUP_COL]
    return X, y, groups


def within_participant_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    task_type: str,
    n_splits_default: int = 5,
) -> Dict[str, Tuple[float, float]]:
    X_all, y_all, groups_all = prepare_subset(df, feature_cols)

    # Determine column types once on the full subset
    numeric_features = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(X_all[c])
    ]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    per_fold_metrics: List[Dict[str, float]] = []

    groups_dict = groups_all.groupby(groups_all).groups

    for participant_id, idx in tqdm(
        groups_dict.items(),
        total=len(groups_dict),
        desc="  Within-participant CV: participants",
        leave=False,
    ):
        idx = np.asarray(list(idx))
        if idx.size < 3:
            # Too few samples for meaningful CV
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

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_p, y_p), start=1):
            model = build_pipeline(
                numeric_features, categorical_features, task_type, RANDOM_STATE
            )
            X_train, X_test = X_p.iloc[train_idx], X_p.iloc[test_idx]
            y_train, y_test = y_p[train_idx], y_p[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred, task_type)
            per_fold_metrics.append(metrics)

    return summarize_metrics(per_fold_metrics)


def cross_participant_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    task_type: str,
) -> Dict[str, Tuple[float, float]]:
    X, y, groups = prepare_subset(df, feature_cols)

    numeric_features = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])
    ]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    logo = LeaveOneGroupOut()
    per_fold_metrics: List[Dict[str, float]] = []
    progress = tqdm(
        total=groups.nunique(),
        desc="  Cross-participant CV: participants",
        leave=False,
    )

    for fold_id, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups), start=1
    ):
        model = build_pipeline(
            numeric_features, categorical_features, task_type, RANDOM_STATE
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, task_type)
        per_fold_metrics.append(metrics)
        progress.update(1)

    progress.close()
    return summarize_metrics(per_fold_metrics)


def run_experiments() -> None:
    df = load_data(DATA_PATH)
    print(f"Loaded data from '{DATA_PATH}' with shape {df.shape}")

    for group_name, raw_features in tqdm(
        FEATURE_GROUPS.items(),
        total=len(FEATURE_GROUPS),
        desc="Feature groups",
    ):
        existing_features = [f for f in raw_features if f in df.columns]
        missing_features = sorted(set(raw_features) - set(existing_features))

        print("\n" + "=" * 80)
        print(f"Feature group: {group_name}")
        print(f"  Requested features: {len(raw_features)}")
        print(f"  Found in data:      {len(existing_features)}")
        print(f"  Missing in data:    {len(missing_features)}")
        if missing_features:
            print("  (Missing feature names will be ignored.)")

        if not existing_features:
            print("  No existing features for this group, skipping.")
            continue

        _, y, _ = prepare_subset(df, existing_features)
        task_type = infer_task_type(y)
        print(f"  Detected task type for '{TARGET_COL}': {task_type}")

        wp_summary = within_participant_cv(df, existing_features, task_type)
        cp_summary = cross_participant_cv(df, existing_features, task_type)

        if not wp_summary and not cp_summary:
            print("  Not enough data for the requested CV schemes.")
            continue

        def print_summary(title: str, summary: Dict[str, Tuple[float, float]]) -> None:
            print(f"  {title}:")
            if not summary:
                print("    (no valid folds)")
                return
            for metric_name, (mean_val, std_val) in summary.items():
                print(f"    {metric_name:10s}: {mean_val:6.3f} ± {std_val:6.3f}")

        print_summary("Within-participant CV", wp_summary)
        print_summary("Cross-participant CV (Leave-One-Participant-Out)", cp_summary)


def main() -> None:
    run_experiments()


if __name__ == "__main__":
    main()

