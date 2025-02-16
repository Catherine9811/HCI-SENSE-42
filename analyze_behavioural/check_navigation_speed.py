import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser


def total_movements(x, y):
    if len(x) < 2 or len(y) < 2:
        return 0  # Not enough points to calculate movement

    # Compute total trajectory distance (sum of Euclidean distances between consecutive points)
    total_distance = sum(np.hypot(np.diff(x), np.diff(y)))

    # Avoid division by zero
    return total_distance


file_path = r"../data/001_explorer_2025-02-15_15h23.13.921.psydat"
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
    # ('Closing Window', 'window_close_mouse.started'),
    ('Opening Folder', 'stimuli_reaction_mouse_movement.started'),
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
        y_values.append(total_movements(x, y) / entry[key][-1])
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
        plt.errorbar(x_means[style], y_means[style], yerr=y_errors[style], fmt='o', capsize=5, label=style,
                     # xerr=x_errors
                     )


# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Moving Speed (unit/second)")
plt.title("Mouse Moving Speed Over Time")
plt.legend()

# Show the plot
plt.show()

