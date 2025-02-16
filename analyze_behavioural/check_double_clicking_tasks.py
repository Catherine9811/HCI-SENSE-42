import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser

file_path = r"../data/001_explorer_2025-02-15_15h23.13.921.psydat"
parser = DataParser(file_path)
print(parser)

# Create the plot
plt.figure(figsize=(6, 4))
# Apply paper-style settings
sns.set_theme(style="whitegrid", context="paper")

task_name = 'Double Click Offset'
task_key = 'stimuli_reaction_mouse_movement.started'
task = parser[task_key]

# Extracting values for plotting
x_values = [entry[task_key] for entry in task]
y_values = []
for entry in task:
    target_location = entry["stimuli_reaction_stimuli_location"]
    last_mouse_location_x = entry[task_key.replace(".started", ".x")][-1]
    last_mouse_location_y = entry[task_key.replace(".started", ".y")][-1]
    distance = np.linalg.norm(np.array([
        target_location[0] - last_mouse_location_x,
        target_location[1] - last_mouse_location_y
    ]))
    y_values.append(distance)
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
plt.title("Accuracy in Double Clicking Tasks Over Time")
plt.legend()

# Show the plot
plt.show()

