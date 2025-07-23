# import numpy as np
# import pickle
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# import addcopyfighandler
# from collections import defaultdict
# from data_parser import DataParser
#
#
# # Use Arial for all text
# mpl.rcParams['font.family'] = 'Arial'
# mpl.rcParams['font.size'] = 14               # base font size
# mpl.rcParams['axes.titlesize'] = 16          # title font size
# mpl.rcParams['axes.labelsize'] = 14          # axis label font size
# mpl.rcParams['xtick.labelsize'] = 12         # x-tick label size
# mpl.rcParams['ytick.labelsize'] = 12         # y-tick label size
# mpl.rcParams['legend.fontsize'] = 12         # legend font size
#
# file_path = r"../data/002_explorer_2025-02-28_14h43.29.510.psydat"
# parser = DataParser(file_path)
# print(parser)
#
# # Apply paper-style settings
# sns.set_theme(style="whitegrid", context="paper")
#
# # Get OS details
# stylizer = parser["operating_system_style"]
# style_mapping = {
#     entry["trials.thisN"]: entry["operating_system_style"]
#     for entry in stylizer if 'trials.thisN' in entry
# }
#
# mouse_pattern = r"^.*\.time$"
#
#
# mouse_tasks = parser.match_entry(mouse_pattern)
#
# # Extracting values for plotting
# x_values = defaultdict(list)
# y_values = defaultdict(list)
#
# for key_name, entries in mouse_tasks.items():
#     x_name = key_name.replace(".time", ".x")
#     y_name = key_name.replace(".time", ".y")
#     for entry in entries:
#         if "trials.thisN" not in entry:
#             continue
#         environment = style_mapping[entry["trials.thisN"]]
#         x_values[environment].extend(entry[x_name])
#         y_values[environment].extend(entry[y_name])
#
# # Plot the heatmap
# fig, axes = plt.subplots(2, 1)
#
# for style, ax in zip(x_values, axes):
#     # Create a 2D histogram for heatmap
#     # heatmap, xedges, yedges = np.histogram2d(x_values[style], y_values[style], range=[[-1, 1], [-1, 1]], bins=(10, 20))
#     #
#     # sns.heatmap(heatmap.T, cmap="Reds", annot=False, linewidths=0.5, ax=ax)
#     ax.scatter(x_values[style], y_values[style], s=10, alpha=0.002)
#     # Labels and title
#     ax.set_xlabel("Screen Width")
#     ax.set_ylabel("Screen Height")
#     ax.set_title(f"Mouse Navigation Heatmap ({style})")
#
# # Show the plot
# plt.show()
#


import os
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from data_parser import DataParser
from scipy.ndimage import gaussian_filter
from data_definition import psydat_files

# Set figure style for paper
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'savefig.dpi': 300,
})

sns.set_theme(style="whitegrid", context="paper")

# Load and parse the data
for file_path in tqdm(psydat_files):
    parser = DataParser(f"../data/{file_path}")

    # Get OS mapping for trials
    stylizer = parser["operating_system_style"]
    style_mapping = {
        entry["trials.thisN"]: entry["operating_system_style"]
        for entry in stylizer if 'trials.thisN' in entry
    }

    # Match mouse movement data
    mouse_pattern = r"^.*\.time$"
    mouse_tasks = parser.match_entry(mouse_pattern)

    # Organize x/y coordinates by environment
    mouse_data = defaultdict(lambda: {"x": [], "y": []})

    for key_name, entries in mouse_tasks.items():
        x_name = key_name.replace(".time", ".x")
        y_name = key_name.replace(".time", ".y")
        for entry in entries:
            if "trials.thisN" not in entry:
                continue
            env = style_mapping[entry["trials.thisN"]]
            mouse_data[env]["x"].extend(entry[x_name])
            mouse_data[env]["y"].extend(entry[y_name])

# Plotting
n_envs = len(mouse_data)
fig, axes = plt.subplots(1, n_envs, figsize=(10 * n_envs, 4), constrained_layout=True)

if n_envs == 1:
    axes = [axes]  # ensure iterable


def norm(data):
    log_data = np.log(data + 1.0)
    return (log_data - log_data.min()) / (log_data.max() - log_data.min())


mapping = {
    "windows": "Trailing Layout",
    "mac": "Leading Layout"
}


for ax, (env, coords) in zip(axes, mouse_data.items()):
    x = np.array(coords["x"])
    y = np.array(coords["y"])

    # KDE-based density heatmap
    if len(x) > 1:
        heatmap, xedges, yedges = np.histogram2d(x, -y, bins=128, range=[[-1, 1], [-1, 1]])
        # heatmap = gaussian_filter(heatmap, sigma=1)

        sns.heatmap(
            norm(heatmap.T), cmap="magma_r", cbar=False, ax=ax,
            xticklabels=False, yticklabels=False
        )

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='magma_r', norm=plt.Normalize(vmin=norm(heatmap.T).min(), vmax=norm(heatmap.T).max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.02)
        cbar.set_label(f"Normalized Mouse Heatmap ({mapping[env]})", fontsize=12, rotation=270, labelpad=16)

        # ax.set_title(f"Mouse Heatmap under {mapping[env]}")
        # ax.set_xlabel("Screen X")
        # ax.set_ylabel("Screen Y")
        ax.set_aspect(1080 / 1728)

        # Optional colorbar
        # cbar = fig.colorbar(scatter, ax=ax)
        # cbar.set_label("Density")

# Save to figures folder
output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)
fig_path = os.path.join(output_folder, "mouse_movement_heatmap.png")
plt.savefig(fig_path, bbox_inches='tight')

# plt.show()
