from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from prediction.alertness.predict_combined_alertness_lgbm_binary_participant_iqr_scaled import (
    BINARY_TARGET_COL,
    GROUP_COL,
    SLEEPINESS_COL,
    TLX_COL,
    create_alertness_binary_target_per_participant,
    load_data,
)
from prediction.alertness.predict_proxy_error_lgbm_binary import (
    PROXY_TARGET_COL,
    create_proxy_error_binary_target_per_participant,
)
from prediction.alertness.shared_config import DATA_PATH, PROXY_ERROR_COLS


BASE_DIR = Path(__file__).resolve().parent
LOWER_PERCENTILE = 0.33
UPPER_PERCENTILE = 0.66
_KAPPA_THRESHOLD = 0.15


def main() -> None:
    df = load_data(DATA_PATH)
    print(f"Loaded data from '{DATA_PATH}' with shape {df.shape}")

    # Build alertness binary + continuous score
    df_both = create_alertness_binary_target_per_participant(
        df=df,
        group_col=GROUP_COL,
        lower_percentile=LOWER_PERCENTILE,
        upper_percentile=UPPER_PERCENTILE,
        new_col=BINARY_TARGET_COL,
    )
    alertness_score_col = "__alertness_score__"
    df_both[alertness_score_col] = df_both[TLX_COL] + df_both[SLEEPINESS_COL]

    # Build proxy binary + continuous composite
    df_both = create_proxy_error_binary_target_per_participant(
        df=df_both,
        group_col=GROUP_COL,
        proxy_cols=PROXY_ERROR_COLS,
        lower_percentile=LOWER_PERCENTILE,
        upper_percentile=UPPER_PERCENTILE,
        new_col=PROXY_TARGET_COL,
    )
    # Recompute composite after the call (the function drops it internally)
    available_proxy = [c for c in PROXY_ERROR_COLS if c in df_both.columns]
    proxy_composite_col = "__proxy_composite__"
    df_both[proxy_composite_col] = df_both[available_proxy].mean(axis=1)

    # ── 1. Per-participant Spearman r ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. Per-participant Spearman r  (composite error score vs alertness score)")
    records = []
    for pid, grp in df_both.groupby(GROUP_COL):
        a = grp[alertness_score_col].dropna()
        b = grp[proxy_composite_col].loc[a.index].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 4:
            continue
        r, p = stats.spearmanr(a.loc[common], b.loc[common])
        records.append({"participant": pid, "spearman_r": r, "p_value": p, "n": len(common)})

    if not records:
        print("  Not enough data for per-participant Spearman analysis.")
    else:
        spearman_df = pd.DataFrame(records)
        print(f"  Participants analysed (>=4 obs): {len(spearman_df)}")
        print(f"  Mean Spearman r  : {spearman_df['spearman_r'].mean():.3f}")
        print(f"  Median Spearman r: {spearman_df['spearman_r'].median():.3f}")
        print(f"  Std Spearman r   : {spearman_df['spearman_r'].std():.3f}")
        sig = spearman_df[spearman_df["p_value"] < 0.05]
        print(f"  Significant (p<0.05): {len(sig)}/{len(spearman_df)} participants")

        output_dir = BASE_DIR / "processed_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        spearman_df.to_csv(output_dir / "proxy_alertness_spearman_per_participant.csv", index=False)
        print(f"  Saved: {output_dir / 'proxy_alertness_spearman_per_participant.csv'}")

    # ── 2. Binary label agreement ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. Binary label agreement  (alertness_binary vs error_proxy_binary)")
    df_binary = df_both[[BINARY_TARGET_COL, PROXY_TARGET_COL]].dropna()
    print(f"  Rows with both labels: {len(df_binary)}")

    if len(df_binary) < 2:
        print("  Not enough paired labels for kappa computation.")
        return

    a_labels = df_binary[BINARY_TARGET_COL].astype(int)
    p_labels = df_binary[PROXY_TARGET_COL].astype(int)

    kappa = cohen_kappa_score(a_labels, p_labels)
    agreement = float(np.mean(a_labels.to_numpy() == p_labels.to_numpy()))

    print(f"  Cohen's kappa    : {kappa:.3f}")
    print(f"  Percent agreement: {agreement:.3f}")

    ct = pd.crosstab(
        a_labels,
        p_labels,
        rownames=["alertness_binary"],
        colnames=["error_proxy_binary"],
    )
    print("\n  Cross-tabulation:")
    print(ct.to_string())

    if kappa >= _KAPPA_THRESHOLD:
        print(
            f"\n  Interpretation: kappa={kappa:.3f} >= {_KAPPA_THRESHOLD} "
            "→ proxy is a meaningful stand-in for alertness (objective alertness detector)."
        )
    else:
        print(
            f"\n  Interpretation: kappa={kappa:.3f} < {_KAPPA_THRESHOLD} "
            "→ proxy captures complementary behavioral degradation. "
            "Frame as 'behavioral performance predictor correlated with alertness'."
        )

    output_dir = BASE_DIR / "processed_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        "metric": ["cohen_kappa", "percent_agreement",
                   "mean_spearman_r", "median_spearman_r"],
        "value": [
            kappa,
            agreement,
            pd.DataFrame(records)["spearman_r"].mean() if records else float("nan"),
            pd.DataFrame(records)["spearman_r"].median() if records else float("nan"),
        ],
    })
    summary.to_csv(output_dir / "proxy_alertness_correlation_summary.csv", index=False)
    print(f"\n  Saved summary: {output_dir / 'proxy_alertness_correlation_summary.csv'}")


if __name__ == "__main__":
    main()
