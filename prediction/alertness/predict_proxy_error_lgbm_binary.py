from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from prediction.alertness.predict_combined_alertness_lgbm_binary_participant_iqr_scaled import (
    GROUP_COL,
    RANDOM_STATE,
    build_lgbm_pipeline,
    fit_participant_iqr_scaler,
    load_data,
    prepare_subset_with_target,
    transform_with_participant_iqr_scaler,
    within_participant_cv_binary_lgbm,
    cross_participant_cv_binary_lgbm,
)
from prediction.alertness.shared_config import (
    DATA_PATH,
    PROXY_ERROR_COLS,
    PROXY_FEATURE_SET_A,
    PROXY_FEATURE_SET_B,
    PROXY_FEATURE_SET_C,
    PROXY_FEATURE_SET_D,
    PROXY_FEATURE_SET_E,
)


BASE_DIR = Path(__file__).resolve().parent
PROXY_TARGET_COL = "error_proxy_binary"


def create_proxy_error_binary_target_per_participant(
    df: pd.DataFrame,
    group_col: str,
    proxy_cols: List[str],
    lower_percentile: float,
    upper_percentile: float,
    new_col: str = PROXY_TARGET_COL,
) -> pd.DataFrame:
    """Build a binary target from a composite behavioral error score.

    Composite = unweighted mean of proxy_cols (_mean error columns only).
    Label 1 = high errors = low alertness state (same semantics as alertness_binary=1).
    """
    available = [c for c in proxy_cols if c in df.columns]
    if not available:
        raise ValueError(f"None of proxy_cols found in dataframe: {proxy_cols}")

    df = df.copy()
    composite_col = "__proxy_composite__"
    df[composite_col] = df[available].mean(axis=1)

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        vals = g[composite_col].dropna().to_numpy()
        if vals.size == 0:
            g[new_col] = np.nan
            return g
        lower_th = np.quantile(vals, lower_percentile)
        upper_th = np.quantile(vals, upper_percentile)
        new_vals = np.full(g.shape[0], np.nan)
        new_vals[g[composite_col] <= lower_th] = 0
        new_vals[g[composite_col] >= upper_th] = 1
        g[new_col] = new_vals
        return g

    df_out = df.groupby(group_col, group_keys=False).apply(_per_group)
    return df_out.drop(columns=[composite_col])


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


def run_proxy_error_experiments(
    lower_percentile: float = 0.33,
    upper_percentile: float = 0.66,
) -> None:
    df = load_data(DATA_PATH)
    print(f"Loaded data from '{DATA_PATH}' with shape {df.shape}")

    # Leakage check: proxy columns must not appear in any feature set
    for name, fset in [
        ("A", PROXY_FEATURE_SET_A),
        ("B", PROXY_FEATURE_SET_B),
        ("C", PROXY_FEATURE_SET_C),
        ("D", PROXY_FEATURE_SET_D),
        ("E", PROXY_FEATURE_SET_E),
    ]:
        leaked = set(PROXY_ERROR_COLS) & set(fset)
        assert not leaked, f"Leakage in set {name}: {leaked}"
    print(f"Leakage check passed ({len(PROXY_ERROR_COLS)} proxy columns not in any feature set).")

    df_proxy = create_proxy_error_binary_target_per_participant(
        df=df,
        group_col=GROUP_COL,
        proxy_cols=PROXY_ERROR_COLS,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        new_col=PROXY_TARGET_COL,
    )

    balance = df_proxy[PROXY_TARGET_COL].value_counts(normalize=True).sort_index()
    print(f"Proxy target class balance: {balance.to_dict()}")

    feature_sets = {
        "A — physiological + behavioural": PROXY_FEATURE_SET_A,
        "B — traits + sleep + engagement": PROXY_FEATURE_SET_B,
        "C — all non-HCI (A ∪ B)": PROXY_FEATURE_SET_C,
        "D — HCI (interaction_hci excl. proxy cols)": PROXY_FEATURE_SET_D,
        "E — all safe features (A ∪ B ∪ D)": PROXY_FEATURE_SET_E,
    }

    summary_rows = []
    for label, fset in feature_sets.items():
        existing = [f for f in fset if f in df_proxy.columns]

        print("\n" + "=" * 80)
        print(f"Feature set {label}")
        print(f"  Features available: {len(existing)} / {len(fset)}")

        if not existing:
            print("  No features available, skipping.")
            continue

        wp, wp_y_true, wp_y_score = within_participant_cv_binary_lgbm(
            df_proxy, existing, target_col=PROXY_TARGET_COL, group_col=GROUP_COL
        )
        cp, cp_y_true, cp_y_score = cross_participant_cv_binary_lgbm(
            df_proxy, existing, target_col=PROXY_TARGET_COL, group_col=GROUP_COL
        )

        wp_pr_auc = _compute_pr_auc(wp_y_true, wp_y_score)
        cp_pr_auc = _compute_pr_auc(cp_y_true, cp_y_score)
        wp_baseline = _compute_pr_baseline(wp_y_true)
        cp_baseline = _compute_pr_baseline(cp_y_true)

        _print_result_block("Within-participant CV", wp)
        _print_result_block("Cross-participant CV (Leave-One-Participant-Out)", cp)
        print(f"  PR-AUC (within): {wp_pr_auc:.3f}  baseline: {wp_baseline:.3f}")
        print(f"  PR-AUC (cross):  {cp_pr_auc:.3f}  baseline: {cp_baseline:.3f}")

        summary_rows.append({
            "feature_set": label,
            "n_features": len(existing),
            "within_precision": wp.get("precision", float("nan")),
            "within_recall": wp.get("recall", float("nan")),
            "within_f1": wp.get("f1", float("nan")),
            "within_pr_auc": wp_pr_auc,
            "within_baseline": wp_baseline,
            "cross_precision": cp.get("precision", float("nan")),
            "cross_recall": cp.get("recall", float("nan")),
            "cross_f1": cp.get("f1", float("nan")),
            "cross_pr_auc": cp_pr_auc,
            "cross_baseline": cp_baseline,
        })

    if summary_rows:
        print("\n" + "=" * 80)
        print("PROXY TARGET — SUMMARY")
        hdr = f"{'Feature set':<38} {'W-F1':>6} {'W-AUC':>7} {'W-Base':>7} {'C-F1':>6} {'C-AUC':>7} {'C-Base':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in summary_rows:
            print(
                f"{r['feature_set']:<38} "
                f"{r['within_f1']:>6.3f} {r['within_pr_auc']:>7.3f} {r['within_baseline']:>7.3f} "
                f"{r['cross_f1']:>6.3f} {r['cross_pr_auc']:>7.3f} {r['cross_baseline']:>7.3f}"
            )

        output_dir = BASE_DIR / "processed_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "proxy_error_lgbm_binary_summary.csv", index=False
        )
        print(f"\nSaved summary: {output_dir / 'proxy_error_lgbm_binary_summary.csv'}")


def main() -> None:
    run_proxy_error_experiments(lower_percentile=0.33, upper_percentile=0.66)


if __name__ == "__main__":
    main()
