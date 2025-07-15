import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from collections import defaultdict

from matplotlib.lines import Line2D
from data_definition import psydat_files
from data_parser import DataParser

# Load data
file_path = rf"../data/{psydat_files[24]}"
parser = DataParser(file_path)
print(parser)

# Apply Seaborn style
plt.figure(figsize=(6, 4))
sns.set_theme(style="whitegrid", context="paper")

# Get OS styles
stylizer = parser["operating_system_style"]
style_mapping = {
    entry["trials.thisN"]: entry["operating_system_style"]
    for entry in stylizer if 'trials.thisN' in entry
}

# Choose task
for task_name, task_key in [
    ('Close Window', 'window_close_mouse.started'),
]:
    task_entries = parser[task_key]

    # Plot preparation
    grouped_data = defaultdict(list)

    for entry in task_entries:
        # Find key with movement data
        candidates = [k for k in entry if k.endswith(".time") and task_key.replace("started", "") in k]
        assert candidates, f"No matching keys for entry: {entry}"
        time_key = candidates[-1]

        # Derive coordinate keys
        x_key = time_key.replace(".time", ".x")
        y_key = time_key.replace(".time", ".y")

        times = np.array(entry[time_key])
        xs = np.array(entry[x_key])
        ys = np.array(entry[y_key])

        if len(xs) == 0 or len(times) == 0:
            continue  # skip if no data

        # Calculate time offsets and distances from the final point
        time_deltas = times[-1] - times[:-1]
        distances = np.sqrt((xs[:-1] - xs[-1])**2 + (ys[:-1] - ys[-1])**2)

        os_style = style_mapping[entry["trials.thisN"]]
        grouped_data[os_style].append((time_deltas, distances))

    # Plot each OS
    for os_style, data in grouped_data.items():
        # Concatenate all time/distance pairs
        all_time_deltas = np.concatenate([d[0] for d in data])
        all_distances = np.concatenate([d[1] for d in data])

        # Bin the time deltas for smoothing (optional)
        bins = np.linspace(0, np.max(all_time_deltas), 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_distances = []
        std_distances = []

        for i in range(len(bins) - 1):
            bin_mask = (all_time_deltas >= bins[i]) & (all_time_deltas < bins[i+1])
            bin_values = all_distances[bin_mask]
            if len(bin_values) > 0:
                mean_distances.append(np.mean(bin_values))
                std_distances.append(np.std(bin_values))
            else:
                mean_distances.append(np.nan)
                std_distances.append(0)

        plt.errorbar(bin_centers, mean_distances, yerr=std_distances, label=os_style, capsize=3)

# Final touches
plt.xlabel("Time before task end (s)")
plt.ylabel("Distance to final point (px)")
plt.title("Mouse Movement Efficiency Across Operating Systems")
plt.legend(title="OS")
plt.tight_layout()
plt.show()
