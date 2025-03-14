import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from collections import defaultdict

from matplotlib.lines import Line2D

from data_parser import DataParser


def effective_movement_ratio(x, y):
    if len(x) < 2 or len(y) < 2:
        return 0  # Not enough points to calculate movement

    # Compute straight-line distance (Euclidean distance between first and last point)
    straight_line_distance = np.hypot(x[-1] - x[0], y[-1] - y[0])

    # Compute total trajectory distance (sum of Euclidean distances between consecutive points)
    total_distance = sum(np.hypot(np.diff(x), np.diff(y)))

    # Avoid division by zero
    return straight_line_distance / total_distance if total_distance > 0 else 0


file_path = r"../data/006_explorer_2025-03-13_19h45.43.154.psydat"
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
    ('Close Window', 'window_close_mouse.started'),
    # ('Open Files', 'file_manager_mouse_homescreen.started'),
    # ('Open Trash', 'trash_bin_mouse_homescreen.started'),
    # ('Open Notes', 'notes_mouse_homescreen.started'),
    # ('Open Browser', 'browser_mouse_homescreen.started')

]:
    typing_task = parser[task_key]

    # typing_task = [entry for entry in typing_task if entry['window_close_target_name'] == 'Minimize']

    # Extracting values for plotting
    x_values = [entry[f"{task_key}"] for entry in typing_task]

    y_values = []

    for entry in typing_task:
        # Get right keys for entry
        candidates = [key for key in entry.keys()
                      if key.endswith(f".{task_key.replace('started', 'time')}") and len(entry[key]) > 0]
        assert len(candidates) > 0, f"{candidates}"
        key = candidates[-1]
        x = entry[key.replace(".time", ".x")]
        y = entry[key.replace(".time", ".y")]
        y_values.append(effective_movement_ratio(x, y))
    g_values = [style_mapping[entry[f"trials.thisN"]] for entry in typing_task]

    # Group data
    grouped_x = defaultdict(list)
    grouped_y = defaultdict(list)

    for g, x, y in zip(g_values, x_values, y_values):
        grouped_x[g].append(x)
        grouped_y[g].append(y)

    means = {style: np.mean(y) for style, y in grouped_y.items()}
    variances = {style: np.var(y) for style, y in grouped_y.items()}

    styles = list(means.keys())
    mean_values = list(means.values())
    variance_values = list(variances.values())

    plt.barh(styles, mean_values, xerr=variance_values, alpha=1.0, label='Mean')

    # for i, (mean, var) in enumerate(zip(mean_values, variance_values)):
    #     plt.vlines(x=styles[i], ymin=mean - var, ymax=mean + var, colors='red',
    #                label='Variance' if i == 0 else "")

# Labels and title
plt.ylabel("Operating System")
plt.xlabel("Effective Moving Percentage (%)")
plt.title("Mouse Navigation Efficiency Over Operating System")

# Show the plot
plt.show()

