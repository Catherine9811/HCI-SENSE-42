import numpy as np
import pickle
import os
import functools
import pandas as pd
import jellyfish
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Polygon
from tqdm import tqdm
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser
from data_definition import psydat_files
from extract_keyboard_handedness_csv import \
    KeyboardPressedDurationExtractor, KeyboardTypingDurationExtractor, KeyboardKeyCountExtractor

# Use Arial for all text
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14               # base font size
mpl.rcParams['axes.titlesize'] = 16          # title font size
mpl.rcParams['axes.labelsize'] = 14          # axis label font size
mpl.rcParams['xtick.labelsize'] = 12         # x-tick label size
mpl.rcParams['ytick.labelsize'] = 12         # y-tick label size
mpl.rcParams['legend.fontsize'] = 12         # legend font size

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


# Define full UK keyboard layout (without number row, but with realistic spacing and key sizes)
keyboard_layout = [
    # Row 1
    [('`', '~', 1.5), ('1', '!'), ('2', '"'), ('3', '£'), ('4', '$'), ('5', '%'), ('6', '^'),
     ('7', '&'), ('8', '*'), ('9', '('), ('0', ')'), ('-', '_'), ('=', '+'), ('Backspace', '', 1.8)],

    # Row 2
    [('Tab', '', 1.8), ('q', 'Q'), ('w', 'W'), ('e', 'E'), ('r', 'R'), ('t', 'T'), ('y', 'Y'),
     ('u', 'U'), ('i', 'I'), ('o', 'O'), ('p', 'P'), ('[', '{'), (']', '}'), ('Enter', '', 1.2)],

    # Row 3
    [('Caps Lock', '', 2), ('a', 'A'), ('s', 'S'), ('d', 'D'), ('f', 'F'), ('g', 'G'), ('h', 'H'),
     ('j', 'J'), ('k', 'K'), ('l', 'L'), (';', ':'), ("'", '@'), ("#", "~"), ('Enter', '')],

    # Row 4
    [('Shift', '', 1.5), ('\\', '|'), ('z', 'Z'), ('x', 'X'), ('c', 'C'), ('v', 'V'), ('b', 'B'), ('n', 'N'),
     ('m', 'M'), (',', '<'), ('.', '>'), ('/', '?'), ('Shift', '', 2.5)],

    # Row 5
    [('LCtrl', '', 1.5), ('', ''), ('LAlt', ''), ('Space', '', 8),
     ('RAlt', ''), ('', ''), ('', ''), ('RCtrl', '')],

    # Row 5
    [('Up', '', 2), ('Down', '', 2), ('Left', '', 2), ('Right', '', 2),
     ('Esc', ''), ('SCR', 'PRT'),
     ('Home', ''), ('End', ''), ('Insert', '', 1.5), ('Delete', '', 1.5)],
]

keyboard_mapping = {
    '`': 'quoteleft',
    '-': 'minus',
    '=': 'equal',
    'enter': 'return',
    '[': 'bracketleft',
    ']': 'bracketright',
    ';': 'semicolon',
    '\'': 'apostrophe',
    '#': 'pound',
    ',': 'comma',
    '.': 'period',
    '/': 'slash',
    '\\': 'backslash',
    'shift': 'lshift',
    'caps lock': 'capslock',
    'esc': 'escape',
    'scr': 'print_screen'
}


# Normalize values for color
def normalize(series):
    if series.nunique() <= 1:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def draw_enter_key(ax, x, y, w1, w2, h, vertical_padding, facecolor, textcolor, edgecolor="black", linewidth=0.6):
    # Define L-shaped points
    points = [
        (x, y),
        (x + w1, y),
        (x + w1, y - 2 * h - vertical_padding),
        (x + w1 - w2, y - 2 * h - vertical_padding),
        (x + w1 - w2, y - h),
        (x, y - h),
    ]

    # Draw polygon
    polygon = Polygon(points, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(polygon)

    # Add label at center
    center_x = x + w1 / 2
    center_y = y - h / 2  # vertically centered in the L
    ax.text(
        center_x, center_y,
        "ENTER",
        ha="center", va="center",
        fontsize=16,
        fontname="Arial",
        weight="medium",
        color=textcolor,
        linespacing=2
    )


# Heatmap plotting per variable
def plot_keyboard_heatmap(df, variable_definitions, save_path="figures"):
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

            fig, axes = plt.subplots(len(group_values), 1, figsize=(18, 6 * len(group_values)))

            if len(group_values) == 1:
                axes = [axes]

            for ax, group_value in zip(axes, group_values):
                group_df = var_df[var_df[group_column] == group_value]
                title = f"{readable} ({group_value})"
                plot_keyboard_heatmap_core(
                    ax=ax,
                    variable=variable,
                    name=name,
                    readable=title,
                    reduce_mode=reduce_mode,
                    var_df=group_df
                )
        else:
            fig, ax = plt.subplots(figsize=(18, 6))
            plot_keyboard_heatmap_core(ax, variable, name, readable, reduce_mode, var_df)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{name}_keyboard_heatmap.png"), dpi=300)
        plt.close()


def plot_keyboard_heatmap_core(ax, variable, name, readable, reduce_mode, var_df):
    if reduce_mode == "average":
        agg = var_df.groupby("key")["value"].mean()
    elif reduce_mode == "sum":
        agg = var_df.groupby("key")["value"].sum()
        agg = np.log(agg + 1e-6)
    elif reduce_mode == "median":
        agg = var_df.groupby("key")["value"].median()
    else:
        raise ValueError(f"Unknown reduce_mode: {reduce_mode}")

    if hasattr(variable, "color_max_clipping"):
        agg = agg.clip(upper=variable.color_max_clipping)

    # Normalize
    agg.index = agg.index.str.lower().str.strip()
    agg_normalized = normalize(agg)

    ax.axis("off")
    ax.set_title(readable, fontsize=18, pad=20)

    safe_margin = 0.05
    total_width = 17
    vertical_padding = 0.02
    height = 1 / 8
    y_offset = 0
    has_enter = False

    ax.set_xlim(0 - safe_margin, total_width + safe_margin)
    ax.set_ylim(0 - (len(keyboard_layout) - 1) * (height + vertical_padding) - safe_margin, height + safe_margin)
    for row in keyboard_layout:
        actual_width = sum([key[2] if len(key) == 3 else 1 for key in row])
        padding = (total_width - actual_width) / (len(row) - 1)
        x_offset = 0
        for key in row:
            # Extract key info
            if len(key) == 3:
                key_label, shift_label, width = key
            else:
                key_label, shift_label = key
                width = 1

            key_lower = key_label.strip().lower()
            heat_val = agg.get(keyboard_mapping.get(key_lower, key_lower), np.nan)
            norm_val = agg_normalized.get(keyboard_mapping.get(key_lower, key_lower), np.nan)

            # Determine color
            color = plt.cm.magma_r(norm_val) if not pd.isna(norm_val) else "lightgrey"

            if pd.isna(norm_val):
                text_color = "black"
            else:
                rgb = color[:3]
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                # Set text color based on luminance
                text_color = 'white' if luminance < 0.5 else 'black'

            if key_label == "Enter":
                if has_enter:
                    continue
                draw_enter_key(ax, x_offset, -y_offset + height, width, 1, height, vertical_padding, color, text_color)
                has_enter = True
            else:
                # Draw rounded key
                rect = FancyBboxPatch(
                    (x_offset, -y_offset),
                    width,
                    height,
                    boxstyle="square,pad=0.0",
                    linewidth=0.6,
                    facecolor=color,
                    edgecolor="black"
                )
                ax.add_patch(rect)

                # Label rendering
                if shift_label and shift_label != key_label.upper():
                    label_text = f"{shift_label}\n{key_label}"
                    fontsize = 16
                else:
                    label_text = key_label.upper()
                    fontsize = 16

                ax.text(
                    x_offset + width / 2,
                    -y_offset + height / 2,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                    weight="medium",
                    fontname="Arial",
                    linespacing=1.5,
                )

            x_offset += width + padding
        y_offset += height + vertical_padding

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='magma_r', norm=plt.Normalize(vmin=agg.min(), vmax=agg.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.02)
    cbar.set_label(readable, fontsize=16, rotation=270, labelpad=20)


if __name__ == '__main__':
    variable_definitions = [
        KeyboardTypingDurationExtractor(),
        KeyboardKeyCountExtractor(),
        KeyboardPressedDurationExtractor()
    ]

    enrollment = os.path.join("..", "..", "data", "participant_enrollment.csv")
    condition_definitions = HandednessExtractor(enrollment)

    # Convert processed dictionary to DataFrame
    df = pd.read_csv(os.path.join("processed_data", "behavioural", f"{len(psydat_files)}-{condition_definitions.name}.csv"))

    plot_keyboard_heatmap(df, variable_definitions)
