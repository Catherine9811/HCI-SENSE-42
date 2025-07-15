import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from collections import defaultdict

from matplotlib.lines import Line2D

from data_parser import DataParser


def mouse_pressed_frames(mouse_pressed, ignore_beginning=True):
    """
        Counts the number of frames where the mouse is pressed (value == 1),
        optionally ignoring leading 1s at the beginning of the array.

        Args:
            mouse_pressed (list or array-like): List of 0s and 1s indicating mouse press state.
            ignore_beginning (bool): Whether to ignore leading 1s at the start.

        Returns:
            int: Number of frames with mouse pressed, considering the ignore_beginning flag.
        """
    if ignore_beginning:
        # Skip leading 1s
        i = 0
        while i < len(mouse_pressed) and mouse_pressed[i] == 1:
            i += 1
        return sum(mouse_pressed[i:])
    else:
        return sum(mouse_pressed)


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
    ('Opening Folder', 'stimuli_reaction_mouse_movement.started'),
    # ('Dragging File', 'stimuli_dragging_mouse.started'),
    # ('Close Window', 'window_close_mouse.started'),
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
        y_values.append(mouse_pressed_frames(left_button))
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
plt.ylabel("Total Frames with Pressed Mouse Key (N)")
plt.title("Mouse Clicking Duration Over Time")

plt.legend()

# Show the plot
plt.show()

