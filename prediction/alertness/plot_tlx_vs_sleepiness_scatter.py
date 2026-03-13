from __future__ import annotations

"""
Create a publication-ready scatter plot:
  - x-axis: sleepiness
  - y-axis: TLX (temporal_demand + mental_demand + effort + frustration - performance)

The figure is saved to prediction/alertness/processed_data/tlx_vs_sleepiness_scatter.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from prediction.alertness.shared_config import DATA_PATH

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "processed_data" / "tlx_vs_sleepiness_scatter.png"

SLEEPINESS_COL = "sleepiness"
TLX_COL = "tlx"
TLX_COMPONENTS = [
    "temporal_demand",
    "mental_demand",
    "effort",
    "frustration",
    "performance",
]


def load_data_with_tlx(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = [SLEEPINESS_COL, *TLX_COMPONENTS]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found in data: {missing_cols}")

    df[TLX_COL] = (
        df["temporal_demand"]
        + df["mental_demand"]
        + df["effort"]
        + df["frustration"]
        - df["performance"]
    )

    return df[[SLEEPINESS_COL, TLX_COL]].dropna().copy()


def create_publication_ready_scatter(df: pd.DataFrame, output_path: Path) -> None:
    # Publication-friendly default style.
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    x = df[SLEEPINESS_COL].to_numpy()
    y = df[TLX_COL].to_numpy()

    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)

    ax.scatter(
        x,
        y,
        s=28,
        alpha=0.65,
        linewidths=0.2,
        edgecolors="white",
        color="#2c7fb8",
    )

    # Fit and draw least-squares trend line.
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#d95f0e", linewidth=2.0, label="Linear fit")

    ax.set_xlabel("Sleepiness")
    ax.set_ylabel("TLX")
    ax.set_title("Relationship Between Sleepiness and TLX")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    annotation = f"n = {len(df)}\nr = {r_val:.3f}\np = {p_val:.2e}"
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    ax.legend(frameon=False, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data_with_tlx(DATA_PATH)
    create_publication_ready_scatter(df, OUTPUT_PATH)
    print(f"Saved plot to: {OUTPUT_PATH}")
    print(f"Rows plotted: {len(df)}")


if __name__ == "__main__":
    main()

