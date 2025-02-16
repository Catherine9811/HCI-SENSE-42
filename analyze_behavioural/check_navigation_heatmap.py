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

# Apply paper-style settings
sns.set_theme(style="whitegrid", context="paper")

# Get OS details
stylizer = parser["operating_system_style"]
style_mapping = {
    entry["trials.thisN"]: entry["operating_system_style"]
    for entry in stylizer if 'trials.thisN' in entry
}

mouse_pattern = r"^.*\.time$"


mouse_tasks = parser.match_entry(mouse_pattern)

# Extracting values for plotting
x_values = defaultdict(list)
y_values = defaultdict(list)

for key_name, entries in mouse_tasks.items():
    x_name = key_name.replace(".time", ".x")
    y_name = key_name.replace(".time", ".y")
    for entry in entries:
        if "trials.thisN" not in entry:
            continue
        environment = style_mapping[entry["trials.thisN"]]
        x_values[environment].extend(entry[x_name])
        y_values[environment].extend(entry[y_name])

# Plot the heatmap
fig, axes = plt.subplots(2, 1)

for style, ax in zip(x_values, axes):
    # Create a 2D histogram for heatmap
    # heatmap, xedges, yedges = np.histogram2d(x_values[style], y_values[style], range=[[-1, 1], [-1, 1]], bins=(10, 20))
    #
    # sns.heatmap(heatmap.T, cmap="Reds", annot=False, linewidths=0.5, ax=ax)
    ax.scatter(x_values[style], y_values[style], s=10, alpha=0.002)
    # Labels and title
    ax.set_xlabel("Screen Width")
    ax.set_ylabel("Screen Height")
    ax.set_title(f"Mouse Navigation Heatmap ({style})")

# Show the plot
plt.show()

