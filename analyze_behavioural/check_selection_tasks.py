import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser

file_path = r"../data/002_explorer_2025-02-28_14h43.29.510.psydat"
parser = DataParser(file_path)
print(parser)

# Create the plot
plt.figure(figsize=(6, 4))
# Apply paper-style settings
sns.set_theme(style="whitegrid", context="paper")

task_name = 'Group Selection'
task_key = 'trash_bin_select_mouse.started'
task = parser[task_key]

# Extracting values for plotting
x_values = [entry[task_key] for entry in task]
y_values = []
for entry in task:
    # Collect location of all folders
    area_start = [1.0, 1.0]
    area_end = [-1.0, -1.0]
    selection_start = entry['selection_start']
    selection_end = entry['selection_end']
    for key_name in entry:
        if key_name.startswith('trash_folder_'):
            area_start[0] = min(area_start[0], entry[key_name][0])
            area_start[1] = min(area_start[1], entry[key_name][1])
            area_end[0] = max(area_end[0], entry[key_name][0])
            area_end[1] = max(area_end[1], entry[key_name][1])
    optimal_area_size = abs((area_end[1] - area_start[1]) * (area_end[0] - area_start[0]))
    actual_area_size = abs((selection_end[1] - selection_start[1]) * (selection_end[0] - selection_start[0]))
    y_values.append(optimal_area_size / actual_area_size)
g_values = [entry[f"trials.thisN"] for entry in task]

# Group data
grouped_x = defaultdict(list)
grouped_y = defaultdict(list)

for g, x, y in zip(g_values, x_values, y_values):
    grouped_x[g].append(x)
    grouped_y[g].append(y)

# Compute mean and standard deviation
x_means = []
x_errors = []
y_means = []
y_errors = []
g_unique = sorted(grouped_x.keys())  # Unique sorted group values

for g in g_unique:
    x_means.append(np.mean(grouped_x[g]))
    x_errors.append(np.std(grouped_x[g]))
    y_means.append(np.mean(grouped_y[g]))
    y_errors.append(np.std(grouped_y[g]))

# Plot error bars
plt.errorbar(x_means, y_means, yerr=y_errors, fmt='o', capsize=5, label=task_name,
             # xerr=x_errors
             )

# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Percentage of Area Covered (unit$^2$)")
plt.title("Accuracy in Group Selection Tasks Over Time")
plt.legend()

# Show the plot
plt.show()

