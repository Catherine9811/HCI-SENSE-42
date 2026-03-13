from __future__ import annotations

"""
SHAP-based feature importance analysis for TLX binary prediction using Gradient Boosting.

This script uses SHAP (SHapley Additive exPlanations) to identify and rank
the most important features contributing to TLX binary classification predictions
when using the mouse_keyboard_traits_sleep_engagement feature group.

TLX (Task Load Index) is calculated as:
  tlx = temporal_demand + mental_demand + effort + frustration - performance
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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


def load_data(path: Path) -> pd.DataFrame:
    """
    Load data and calculate TLX (Task Load Index).
    
    TLX is calculated as: temporal_demand + mental_demand + effort + frustration - performance
    """
    df = pd.read_csv(path)
    
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found in data.")
    
    # Required columns for TLX calculation
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
    
    # Calculate TLX: temporal_demand + mental_demand + effort + frustration - performance
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

    # Use group_keys=False so that the original index/columns are preserved
    # Note: We don't use include_groups=False because we need the group_col in the result
    df_bin = df_bin.groupby(group_col, group_keys=False).apply(_per_group)
    return df_bin


def prepare_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data by selecting features and target, dropping NaNs.
    """
    # Only include group_col if it exists in the dataframe
    cols_to_select = feature_cols + [target_col]
    if group_col in df.columns:
        cols_to_select.append(group_col)
    
    cols = list(dict.fromkeys(cols_to_select))
    subset = df[cols].copy()
    
    # Drop NaNs: use group_col only if it exists
    dropna_subset = [target_col]
    if group_col in subset.columns:
        dropna_subset.append(group_col)
    subset = subset.dropna(subset=dropna_subset)
    subset = subset.reset_index(drop=True)
    
    X = subset[feature_cols]
    y = subset[target_col]
    return X, y


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

    # Handle OneHotEncoder sparse parameter for different sklearn versions
    try:
        # Newer sklearn versions use sparse_output
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn versions use sparse
        try:
            onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            # Very old versions, use default and convert later if needed
            onehot_encoder = OneHotEncoder(handle_unknown="ignore")
    
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot_encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
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


def get_feature_names_after_preprocessing(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
    X_sample: pd.DataFrame,
) -> List[str]:
    """
    Get feature names after preprocessing (including one-hot encoding).
    """
    preprocessor = pipeline.named_steps["preprocess"]
    
    # Get numeric feature names (unchanged)
    numeric_names = list(numeric_features)
    
    # Get categorical feature names after one-hot encoding
    categorical_transformer = preprocessor.named_transformers_["cat"]
    onehot = categorical_transformer.named_steps["onehot"]
    
    # Get feature names from one-hot encoder
    categorical_names = []
    if hasattr(onehot, "get_feature_names_out"):
        # New sklearn API (v1.0+)
        try:
            cat_feature_names = onehot.get_feature_names_out(categorical_features)
            categorical_names = list(cat_feature_names)
        except (TypeError, ValueError):
            # Some versions don't require input, or input format differs
            try:
                cat_feature_names = onehot.get_feature_names_out()
                categorical_names = list(cat_feature_names)
            except Exception:
                # Fallback: construct names manually from categories
                categorical_names = []
                if hasattr(onehot, "categories_"):
                    for i, cat_feat in enumerate(categorical_features):
                        categories = onehot.categories_[i]
                        for cat_val in categories:
                            categorical_names.append(f"{cat_feat}_{cat_val}")
    elif hasattr(onehot, "get_feature_names"):
        # Older sklearn API
        try:
            cat_feature_names = onehot.get_feature_names(categorical_features)
            categorical_names = list(cat_feature_names)
        except Exception:
            # Fallback
            categorical_names = []
            if hasattr(onehot, "categories_"):
                for i, cat_feat in enumerate(categorical_features):
                    categories = onehot.categories_[i]
                    for cat_val in categories:
                        categorical_names.append(f"{cat_feat}_{cat_val}")
    else:
        # Fallback: construct names manually from categories
        categorical_names = []
        if hasattr(onehot, "categories_"):
            for i, cat_feat in enumerate(categorical_features):
                categories = onehot.categories_[i]
                for cat_val in categories:
                    categorical_names.append(f"{cat_feat}_{cat_val}")
    
    return numeric_names + categorical_names


def compute_shap_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    sample_size: int = 100,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute SHAP values and return feature importance rankings.
    
    Note: The pipeline should already be fitted before calling this function.
    
    Returns:
        - Mean absolute SHAP values for each feature
        - Feature names after preprocessing
    """
    # Get the model from the pipeline (should already be fitted)
    model = pipeline.named_steps["model"]
    
    # Transform X to get preprocessed features
    X_transformed = pipeline.named_steps["preprocess"].transform(X)
    
    # Convert sparse matrix to dense if needed
    from scipy.sparse import issparse
    if issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    
    # Convert to DataFrame for easier handling
    feature_names = get_feature_names_after_preprocessing(
        pipeline, numeric_features, categorical_features, X
    )
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # Sample data for SHAP computation (for efficiency)
    if len(X_transformed_df) > sample_size:
        X_sample = X_transformed_df.sample(n=sample_size, random_state=RANDOM_STATE)
    else:
        X_sample = X_transformed_df
    
    # Use TreeExplainer for Gradient Boosting
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values might be a list [values_for_class_0, values_for_class_1]
    # We use the values for class 1 (positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    return mean_abs_shap, feature_names


def _map_shap_feature_to_base_feature(
    shap_feature_name: str,
    numeric_features: List[str],
    categorical_features: List[str],
) -> str:
    """
    Map a post-preprocessing feature name back to its original input feature.

    - Numeric features stay unchanged.
    - One-hot encoded categorical features are expected to be "{col}_{category}",
      so we map them back to "{col}" by prefix matching on known categorical columns.
    """
    if shap_feature_name in numeric_features:
        return shap_feature_name

    for cat_feat in categorical_features:
        prefix = f"{cat_feat}_"
        if shap_feature_name.startswith(prefix) or shap_feature_name == cat_feat:
            return cat_feat

    # Fallback: unknown naming format; keep as-is
    return shap_feature_name


def analyze_feature_importance_shap(
    lower_percentile: float = 1.0 / 3.0,
    upper_percentile: float = 2.0 / 3.0,
    sample_size: int = 100,
) -> None:
    """
    Analyze feature importance using SHAP for TLX binary classification.
    """
    # Load and prepare data
    df = load_data(DATA_PATH)
    print(f"Loaded data from '{DATA_PATH}' with shape {df.shape}")
    print(
        f"Calculated '{ORIGINAL_TARGET_COL}' from temporal_demand + mental_demand + "
        f"effort + frustration - performance"
    )
    
    # Create binary target
    print(
        f"Creating binary target '{BINARY_TARGET_COL}' from '{ORIGINAL_TARGET_COL}' "
        f"using within-participant lower/upper percentiles = "
        f"{lower_percentile:.2f}/{upper_percentile:.2f}"
    )
    
    df_bin = create_binary_target_per_participant(
        df,
        outcome_col=ORIGINAL_TARGET_COL,
        group_col=GROUP_COL,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        new_col=BINARY_TARGET_COL,
    )
    
    # Get features for mouse_keyboard_traits_sleep_engagement group
    group_name = "mouse_keyboard_traits_sleep_engagement"
    if group_name not in FEATURE_GROUPS:
        raise ValueError(f"Feature group '{group_name}' not found in FEATURE_GROUPS.")
    
    raw_features = FEATURE_GROUPS[group_name]
    existing_features = [f for f in raw_features if f in df_bin.columns]
    missing_features = sorted(set(raw_features) - set(existing_features))
    
    print("\n" + "=" * 80)
    print(f"Feature group: {group_name}")
    print(f"  Requested features: {len(raw_features)}")
    print(f"  Found in data:      {len(existing_features)}")
    print(f"  Missing in data:    {len(missing_features)}")
    if missing_features:
        print("  (Missing feature names will be ignored.)")
    
    if not existing_features:
        print("  No existing features for this group, exiting.")
        return
    
    # Prepare data
    X, y = prepare_data(df_bin, existing_features, BINARY_TARGET_COL, GROUP_COL)
    
    print(f"\nData shape after preprocessing: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Check if we have both classes
    if len(y.unique()) < 2:
        print("  Warning: Only one class present in target. Cannot train binary classifier.")
        return
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Identify numeric and categorical features
    numeric_features = [c for c in existing_features if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_features = [c for c in existing_features if c not in numeric_features]
    
    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Build and train pipeline
    print("\nTraining Gradient Boosting model...")
    pipeline = build_gb_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Compute SHAP values
    print("\nComputing SHAP values...")
    mean_abs_shap, feature_names = compute_shap_importance(
        pipeline, X_train, numeric_features, categorical_features, sample_size=sample_size
    )
    
    # Create ranking DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    
    # Print results
    print("\n" + "=" * 80)
    print("Feature Importance Ranking (SHAP - Mean Absolute SHAP Values)")
    print("=" * 80)
    print(f"\nTop {min(20, len(importance_df))} most important features:\n")
    
    for idx, row in importance_df.head(20).iterrows():
        print(f"{row['mean_abs_shap']:8.4f}  {row['feature']}")
    
    # Save results to CSV
    output_path = BASE_DIR / "processed_data" / "tlx_gb_binary_shap_importance.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    # Summarize contribution by FEATURE_GROUP (atomic groups; exclude combined/overall)
    atomic_group_names = [
        g
        for g in FEATURE_GROUPS.keys()
        if g not in {"mouse_keyboard_traits_sleep_engagement", "all_features"}
    ]

    base_feature_to_group: Dict[str, str] = {}
    for g in atomic_group_names:
        for f in FEATURE_GROUPS.get(g, []):
            # Prefer first assignment if ever duplicated
            base_feature_to_group.setdefault(f, g)

    importance_df = importance_df.copy()
    importance_df["base_feature"] = importance_df["feature"].apply(
        lambda s: _map_shap_feature_to_base_feature(s, numeric_features, categorical_features)
    )
    importance_df["feature_group"] = importance_df["base_feature"].map(base_feature_to_group).fillna(
        "unknown"
    )

    group_contrib_df = (
        importance_df.groupby("feature_group", as_index=False)
        .agg(
            n_encoded_features=("feature", "count"),
            n_base_features=("base_feature", pd.Series.nunique),
            sum_mean_abs_shap=("mean_abs_shap", "sum"),
            mean_mean_abs_shap=("mean_abs_shap", "mean"),
        )
        .sort_values("sum_mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    total_shap = float(group_contrib_df["sum_mean_abs_shap"].sum())
    group_contrib_df["share_of_total"] = (
        group_contrib_df["sum_mean_abs_shap"] / total_shap if total_shap > 0 else 0.0
    )

    group_output_path = BASE_DIR / "processed_data" / "tlx_gb_binary_shap_group_contribution.csv"
    group_output_path.parent.mkdir(parents=True, exist_ok=True)
    group_contrib_df.to_csv(group_output_path, index=False)

    print("\n" + "=" * 80)
    print("FEATURE_GROUP Contribution Summary (by sum of mean_abs_shap)")
    print("=" * 80)
    print(group_contrib_df.to_string(index=False))
    print(f"\nGroup contribution results saved to: {group_output_path}")
    
    return importance_df


def main() -> None:
    # Default to keeping bottom/top 33% and discarding middle 33% per participant.
    analyze_feature_importance_shap(
        lower_percentile=1.49 / 3.0,
        upper_percentile=1.51 / 3.0,
        sample_size=100,
    )


if __name__ == "__main__":
    main()
