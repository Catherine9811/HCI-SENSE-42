import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from analyze_respiratory.rip import Resp
from matplotlib.patches import Patch

from scipy.stats import ttest_rel

import addcopyfighandler


def add_significance_annotation(ax, x1, x2, y1, y2, h, p_val, text_offset=0.02):
    """
    Draws significance bars with stars between x1 and x2 on ax at height y.
    h: height of the vertical bar.
    """
    # Determine significance stars
    if p_val < 0.00001:
        stars = f'p < 0.00001 (****) '
    elif p_val < 0.0001:
        stars = f'p < 0.0001 (****)'
    elif p_val < 0.001:
        stars = f'p < 0.001 (***)'
    elif p_val < 0.01:
        stars = f'p = {p_val:.3f} (**)'
    elif p_val < 0.05:
        stars = f'p = {p_val:.3f} (*)'
    else:
        stars = '(n.s.)'

    # Draw horizontal line
    ax.plot([x1, x1, x2, x2], [y1, max(y1, y2) + h, max(y1, y2) + h, y2], color='black', linewidth=1.5)
    # Add stars text
    ax.text((x1 + x2) * 0.5, max(y1, y2) + h + text_offset, stars, font="Arial",
            ha='center', va='bottom', color='black', fontsize=12)


mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

# Caching config
use_cache = True
cache_folder = "data"
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

# Define cache file paths
cache_files = {
    "inh_dur_means": os.path.join(cache_folder, "inh_dur_means.npy"),
    "inh_dur_stds": os.path.join(cache_folder, "inh_dur_stds.npy"),
    "exh_dur_means": os.path.join(cache_folder, "exh_dur_means.npy"),
    "exh_dur_stds": os.path.join(cache_folder, "exh_dur_stds.npy"),
    "group_labels": os.path.join(cache_folder, "group_labels.npy")
}

# Load from cache if available
if use_cache and all(os.path.exists(f) for f in cache_files.values()):
    inh_dur_means = np.load(cache_files["inh_dur_means"])
    inh_dur_stds = np.load(cache_files["inh_dur_stds"])
    exh_dur_means = np.load(cache_files["exh_dur_means"])
    exh_dur_stds = np.load(cache_files["exh_dur_stds"])
    group_labels = np.load(cache_files["group_labels"])
else:
    inh_dur_means = []   # per participant or group
    inh_dur_stds = []
    exh_dur_means = []
    exh_dur_stds = []
    group_labels = []    # e.g. "first 30 min", "last 30 min", or Group A–D

    for pid in tqdm(range(1, 43)):
        try:
            resp = Resp.from_wav(f"../data/Respiration/P{pid:03d}.wav")
            resp.remove_baseline(method='savgol')
            resp.find_cycles(include_holds=True)

            inh_durs = [i.duration() for i in resp.inhalations]
            exh_durs = [i.duration() for i in resp.exhalations]

            # Time window splits (like ECG example)
            first_30 = lambda x: [d.duration() for d in x if d.start_time < 1800]
            last_30  = lambda x: [d.duration() for d in x if d.start_time >= resp.dur - 1800]

            for label, selector in [("first", first_30), ("last", last_30)]:
                inhs = selector(resp.inhalations)
                exhs = selector(resp.exhalations)

                if inhs and exhs:
                    inh_dur_means.append(np.mean(inhs))
                    inh_dur_stds.append(np.std(inhs))
                    exh_dur_means.append(np.mean(exhs))
                    exh_dur_stds.append(np.std(exhs))
                    group_labels.append(label)
        except Exception as e:
            print(f"Skipping P{pid:03d}: {e}")

    # Convert to numpy arrays
    inh_dur_means = np.array(inh_dur_means)
    inh_dur_stds = np.array(inh_dur_stds)
    exh_dur_means = np.array(exh_dur_means)
    exh_dur_stds = np.array(exh_dur_stds)
    group_labels = np.array(group_labels)

    np.save(cache_files["inh_dur_means"], inh_dur_means)
    np.save(cache_files["inh_dur_stds"], inh_dur_stds)
    np.save(cache_files["exh_dur_means"], exh_dur_means)
    np.save(cache_files["exh_dur_stds"], exh_dur_stds)
    np.save(cache_files["group_labels"], group_labels)

# Split first/last
first_mask = group_labels == "first"
last_mask = group_labels == "last"

labels = ['Inhalation', 'Exhalation', 'Inhalation Variability', 'Exhalation Variability']
x = np.arange(len(labels))
width = 0.15

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

# Violinplot 1: durations
vp1 = ax1.violinplot([
    inh_dur_means[first_mask], inh_dur_means[last_mask],
    exh_dur_means[first_mask], exh_dur_means[last_mask]
], positions=[0 - width, 0 + width, 1 - width, 1 + width],
   widths=width, showmedians=True, showextrema=False)

# Violinplot 2: variability (on ax2)
vp2 = ax2.violinplot([
    inh_dur_stds[first_mask], inh_dur_stds[last_mask],
    exh_dur_stds[first_mask], exh_dur_stds[last_mask]
], positions=[2 - width, 2 + width, 3 - width, 3 + width],
   widths=width, showmedians=True, showextrema=False)

# Colors
colors = ['tab:blue', 'tab:orange'] * 2
for i, b in enumerate(vp1['bodies']):
    b.set_facecolor(colors[i])
    b.set_edgecolor('black')
    b.set_alpha(0.7)
    b.set_zorder(3)

for i, b in enumerate(vp2['bodies']):
    b.set_facecolor(colors[i])
    b.set_edgecolor('black')
    b.set_alpha(0.7)
    b.set_zorder(3)

# Medians
vp1['cmedians'].set_color('black')
vp1['cmedians'].set_zorder(4)
vp2['cmedians'].set_color('black')
vp2['cmedians'].set_zorder(4)


for first, last, x1, x2 in zip(
    [
        inh_dur_means[first_mask], exh_dur_means[first_mask], inh_dur_stds[first_mask], exh_dur_stds[first_mask]
    ],
    [
        inh_dur_means[last_mask], exh_dur_means[last_mask], inh_dur_stds[last_mask], exh_dur_stds[last_mask]
    ],
    [
        0 - width, 1 - width, 2 - width, 3 - width
    ],
    [
        0 + width, 1 + width, 2 + width, 3 + width
    ]
):
    max_val = max(first.max(), last.max())
    min_val = min(first.min(), last.min())
    print(last.max())
    t, p = ttest_rel(first, last)
    add_significance_annotation(ax2, x1=x1, x2=x2, y1=first.max()*1.05, y2=last.max()*1.05,
                                h=max_val*0.05, p_val=p, text_offset=0.02*(max_val - min_val))


# Axes
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Duration (s)")
ax1.set_ylim(0.5, 4.6)
ax2.set_ylabel("Variance (s)", rotation=-90, labelpad=15)
ax1.set_xlabel("Respiration Timing Metrics")
# Legend
handles = [
    Patch(facecolor='tab:blue', edgecolor='black', label='first 30 min'),
    Patch(facecolor='tab:orange', edgecolor='black', label='last 30 min')
]
# legend = ax1.legend(handles=handles, loc='upper right')
# legend.get_frame().set_edgecolor('black')  # set solid edge color
# legend.get_frame().set_linewidth(1.0)      # optional: control thickness

ax1.grid(True, linestyle='--', alpha=0.5)
ax1.grid(axis='x', visible=False)  # hide y-axis grid
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
