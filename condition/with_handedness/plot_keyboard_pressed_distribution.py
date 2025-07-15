import numpy as np
import pickle
import os
import functools
import pandas as pd
import jellyfish
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from data_definition import psydat_files
from extract_keyboard_handedness_csv import \
    KeyboardPressedDurationExtractor, KeyboardTypingDurationExtractor, KeyboardKeyCountExtractor

from scipy.stats import f_oneway, levene


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_ASPECT_RATIO = 1920 / 1080


def filter_time_series(times, values):
    return times, values
    # times = np.array(times)
    # values = np.array(values)
    # condition = (times < 2 * 60 * 60)
    # print(f"Time filtered {np.sum(condition)} items.")
    # return times[condition], values[condition]


def filter_nan_indices(processed):
    """Remove all indices where any value in the dictionary contains NaN."""
    # Find indices where any column contains NaN
    nan_indices = {i for values in processed.values() for i, v in enumerate(values) if np.isnan(v)}

    # Keep only the indices that are NOT in nan_indices
    processed = {key: [v for i, v in enumerate(values) if i not in nan_indices] for key, values in processed.items()}

    print(f"{len(set(nan_indices))} filtered.")

    return processed


def to_title_case(s):
    return s.replace("_", " ").title()


def scale_mouse_measurement_to_screen_height(measure):
    if isinstance(measure, tuple):
        return measure[0] * SCREEN_ASPECT_RATIO, measure[1]
    measure[0] *= SCREEN_ASPECT_RATIO
    return measure


def scale_width_measurement_to_screen_height(width):
    return width * SCREEN_ASPECT_RATIO


# Normalize values for color
def normalize(series):
    if series.nunique() <= 1:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


class HandednessExtractor:
    name = "keyboard_handedness"

    def __init__(self, csv_path):
        self.enrollment = pd.read_csv(csv_path)
        self.id_column = "Participant ID"
        self.uh_column = "Handedness"
        # Extract the relevant columns
        data = self.enrollment[[self.id_column, self.uh_column]].dropna()

        # Create a dictionary: {participant_id (int): str}
        self.uh_dict = {
            int(row[self.id_column]): str(row[self.uh_column])
            for _, row in data.iterrows()
        }

    def process(self, participant_id: int):
        return self.uh_dict.get(participant_id, "")


# Heatmap plotting per variable
def plot_keyboard_pressed_distribution(df, variable_definitions, key_definitions, save_path="figures"):
    os.makedirs(save_path, exist_ok=True)

    for variable in variable_definitions:
        name = variable.name
        readable = variable.readable_name
        reduce_mode = variable.reduce_mode

        # Aggregate by key
        var_df = df[df["name"] == name]
        # Filter out extreme values
        var_df = var_df[var_df["value"] <= variable.max_clipping]
        var_df = var_df[var_df["value"] >= 0]

        if hasattr(variable, "group_by"):
            group_column = variable.group_by
            group_values = sorted(var_df[group_column].unique())  # <-- sorted here

            key_column = "key"
            key_values = key_definitions  # <-- sorted here

            fig, axes = plt.subplots(len(group_values), len(key_values), figsize=(18, 6 * len(group_values)))

            if len(group_values) == 1:
                axes = [axes]

            for ax, key_value in zip(axes, key_values):
                for axi, group_value in zip(ax, group_values):
                    title = f"{readable} ({group_value}) [{key_value.upper()}]"
                    plot_keyboard_pressed_distribution_core(
                        ax=axi,
                        variable=variable,
                        name=name,
                        readable=title,
                        reduce_mode=reduce_mode,
                        var_df=var_df[(var_df[key_column] == key_value) & (var_df[group_column] == group_value)]
                    )
            # === One-Way ANOVA + Levene's test per key across groups ===
            print(f"\n--- One-Way ANOVA: {readable} ---")
            for key_value in key_values:
                key_df = var_df[var_df[key_column] == key_value]

                # Compute mean per participant per group
                agg_df = (
                    key_df.groupby(["participant", group_column])["value"]
                    .median()
                    .reset_index()
                )

                # Prepare group data
                group_data = {
                    group: agg_df[agg_df[group_column] == group]["value"].values
                    for group in group_values
                }

                # Filter out groups with < 2 participants
                group_data = {k: v for k, v in group_data.items() if len(v) >= 2}

                if len(group_data) < 2:
                    print(f"Not enough data for ANOVA on key '{key_value.upper()}'. Skipping.")
                    continue

                try:
                    # Run one-way ANOVA
                    stat, pval = f_oneway(*group_data.values())

                    # Run Levene's test for homogeneity of variance
                    levene_stat, levene_p = levene(*group_data.values())

                    print(f"\nKey: {key_value.upper()}")
                    print("Groups:", list(group_data.keys()))
                    print(f"ANOVA:  F = {stat:.3f}, p = {pval:.4f}")
                    print(f"Levene: W = {levene_stat:.3f}, p = {levene_p:.4f} ", end="")
                    if levene_p < 0.05:
                        print("⚠️ Variance differs significantly across groups.")
                    else:
                        print("✓ Homogeneity of variance assumed.")

                except Exception as e:
                    print(f"ANOVA failed for key '{key_value.upper()}': {e}")
        else:
            key_column = "key"
            key_values = sorted(key_definitions)  # <-- sorted here

            fig, axes = plt.subplots(len(key_values), 1, figsize=(18, 6 * len(key_values)))

            if len(key_values) == 1:
                axes = [axes]

            for ax, key_value in zip(axes, key_values):
                title = f"{readable} [{key_value.upper()}]"
                plot_keyboard_pressed_distribution_core(
                    ax=ax,
                    variable=variable,
                    name=name,
                    readable=title,
                    reduce_mode=reduce_mode,
                    var_df=var_df[var_df[key_column] == key_value]
                )
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{name}_keyboard_pressed_distribution.png"), dpi=300)
        plt.close()


def plot_keyboard_pressed_distribution_core(ax, variable, name, readable, reduce_mode, var_df):
    values = var_df["value"]

    if hasattr(variable, "color_max_clipping"):
        values = values[values <= variable.color_max_clipping]

    # Plot histogram data (but don't draw it yet)
    counts, bins, patches = ax.hist(values, bins=11, edgecolor='black')

    # Normalize counts for coloring
    norm = Normalize(vmin=0, vmax=max(counts))
    cmap = plt.get_cmap('copper_r')

    # Color bins based on height
    for count, patch in zip(counts, patches):
        color = cmap(norm(count))
        patch.set_facecolor(color)

    # Title and labels
    ax.set_title(readable, fontsize=18, pad=20)
    ax.set_xlabel(name, fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

    # Optional: add colorbar for reference
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Bin Frequency')


if __name__ == '__main__':
    variable_definitions = [
        KeyboardPressedDurationExtractor()
    ]
    variable_definitions[0].color_max_clipping = 0.20
    key_definitions = [
        "q", "r", "space", "u", "p"
    ]

    enrollment = os.path.join("..", "..", "data", "participant_enrollment.csv")
    condition_definitions = HandednessExtractor(enrollment)

    # Convert processed dictionary to DataFrame
    df = pd.read_csv(os.path.join("processed_data", "behavioural", f"{len(psydat_files)}-{condition_definitions.name}.csv"))

    plot_keyboard_pressed_distribution(df, variable_definitions, key_definitions)



