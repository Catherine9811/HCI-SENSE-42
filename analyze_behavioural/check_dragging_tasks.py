import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser

file_path = r"../data/006_explorer_2025-03-13_19h45.43.154.psydat"
parser = DataParser(file_path)
print(parser)

# Create the plot
plt.figure(figsize=(6, 4))
# Apply paper-style settings
sns.set_theme(style="whitegrid", context="paper")

task_name = 'Drag Offset'
task_key = 'stimuli_dragging_mouse.started'
task = parser[task_key]

# Extracting values for plotting
x_values = [entry[task_key] for entry in task]
y_values = [np.linalg.norm(entry["last_clicked_offset"]) for entry in task]
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
plt.ylabel("Distance (unit)")
plt.title("Accuracy in Dragging Tasks Over Time")
plt.legend()

# Show the plot
plt.show()

