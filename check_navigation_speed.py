import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser

file_path = r"data\094814_explorer_2025-01-31_13h44.24.341.psydat"
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
    for entry in stylizer
}

for task_name, task_key in [
    ('Closing Window', 'window_close_mouse.started'),
]:
    typing_task = parser[task_key]

    # Extracting values for plotting
    x_values = [entry[f"{task_key}"] for entry in typing_task]

    y_values = []

    for entry in typing_task:
        # Get right keys for entry
        candidates = [key for key in entry.keys() if key.endswith(".window_close_mouse.time") and len(entry[key]) > 0]
        assert len(candidates) > 0
        y_values.append(entry[candidates[-1]][-1])
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
plt.ylabel("Time Spent (seconds)")
plt.title("Time Used in Closing Windows Over Time")
plt.legend()

# Show the plot
plt.show()

