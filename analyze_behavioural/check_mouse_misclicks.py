import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from collections import defaultdict

from matplotlib.lines import Line2D

from data_parser import DataParser


def mouse_misclicked_times(mouse_pressed, ignore_beginning=True, ignore_ending=True):
    """
    Counts the number of groups of continuous 1s in the mouse_pressed sequence,
    optionally ignoring leading and/or trailing 1s.

    Args:
        mouse_pressed (list or array-like): List of 0s and 1s indicating mouse press state.
        ignore_beginning (bool): Whether to ignore leading 1s at the start.
        ignore_ending (bool): Whether to ignore trailing 1s at the end.

    Returns:
        int: Number of groups of consecutive 1s (mouse misclicks), considering options.
    """
    n = len(mouse_pressed)
    start = 0
    end = n

    # Ignore leading 1s
    if ignore_beginning:
        while start < n and mouse_pressed[start] == 1:
            start += 1

    # Ignore trailing 1s
    if ignore_ending:
        while end > start and mouse_pressed[end - 1] == 1:
            end -= 1

    # Count transitions from 0 to 1
    in_group = False
    group_count = 0
    for i in range(start, end):
        if mouse_pressed[i] == 1:
            if not in_group:
                group_count += 1
                in_group = True
        else:
            in_group = False

    return group_count


file_path = r"../data/007_explorer_2025-03-14_14h12.43.192.psydat"
parser = DataParser(file_path)
print(parser)

# Create the plot
plt.figure(figsize=(6, 4))
# Apply paper-style settings
sns.set_theme(style="whitegrid", context="paper")

# Get OS details
stylizer = parser["operating_system_style"]
style_mapping = {
    entry["trials.thisN"]: entry["operating_system_style"]
    for entry in stylizer if 'trials.thisN' in entry
}

for task_name, task_key in [
    # ('Close Window', 'window_close_mouse.started'),
    # ('Opening Folder', 'stimuli_reaction_mouse_movement.started'),
    ('Dragging File', 'stimuli_dragging_mouse.started'),
    # ('Open Files', 'file_manager_mouse_homescreen.started'),
    # ('Open Trash', 'trash_bin_mouse_homescreen.started'),
    # ('Open Notes', 'notes_mouse_homescreen.started'),
    # ('Open Browser', 'browser_mouse_homescreen.started')

]:
    typing_task = parser[task_key]

    # Extracting values for plotting
    x_values = [entry[f"{task_key}"] for entry in typing_task]

    y_values = []

    for entry in typing_task:
        # Get right keys for entry
        candidates = [key for key in entry.keys()
                      if key.endswith(f".{task_key.replace('started', 'time')}") and len(entry[key]) > 0]
        assert len(candidates) > 0, f"{candidates}"
        key = candidates[-1]
        left_button = entry[key.replace(".time", ".leftButton")]
        y_values.append(mouse_misclicked_times(left_button))
    g_values = [entry[f"trials.thisN"] for entry in typing_task]

    # Group data
    grouped_x = defaultdict(list)
    grouped_y = defaultdict(list)

    for g, x, y in zip(g_values, x_values, y_values):
        grouped_x[g].append(x)
        grouped_y[g].append(y)

    # Compute mean and standard deviation
    x_means = defaultdict(list)
    x_errors = defaultdict(list)
    y_means = defaultdict(list)
    y_errors = defaultdict(list)
    g_unique = sorted(grouped_x.keys())  # Unique sorted group values

    for g in g_unique:
        x_means[style_mapping[g]].append(np.mean(grouped_x[g]))
        x_errors[style_mapping[g]].append(np.std(grouped_x[g]))
        y_means[style_mapping[g]].append(np.mean(grouped_y[g]))
        y_errors[style_mapping[g]].append(np.std(grouped_y[g]))

    for style in x_means:
        # Plot error bars
        plt.errorbar(x_means[style], y_means[style], yerr=y_errors[style], fmt='o', capsize=5,
                     label=f"{task_name} {style}"
                     # xerr=x_errors
                     )


# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Times of Unintended Click (N)")
plt.title("Unintended Clicks Over Time")

plt.legend()

# Show the plot
plt.show()

