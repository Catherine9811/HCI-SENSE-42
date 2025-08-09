import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from analyze_respiratory.rip import Resp
from matplotlib.patches import Patch
import addcopyfighandler

# --------------------
# Matplotlib styles
# --------------------
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

# --------------------
# Cache settings
# --------------------
use_cache = True
cache_folder = "data"
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

cache_files = {
    "inh_dur_means": os.path.join(cache_folder, "inh_dur_means_all.npy"),
    "inh_dur_stds": os.path.join(cache_folder, "inh_dur_stds_all.npy"),
    "exh_dur_means": os.path.join(cache_folder, "exh_dur_means_all.npy"),
    "exh_dur_stds": os.path.join(cache_folder, "exh_dur_stds_all.npy"),
}

# --------------------
# Load or compute data
# --------------------
if use_cache and all(os.path.exists(f) for f in cache_files.values()):
    inh_dur_means = np.load(cache_files["inh_dur_means"])
    inh_dur_stds = np.load(cache_files["inh_dur_stds"])
    exh_dur_means = np.load(cache_files["exh_dur_means"])
    exh_dur_stds = np.load(cache_files["exh_dur_stds"])
else:
    inh_dur_means = []
    inh_dur_stds = []
    exh_dur_means = []
    exh_dur_stds = []

    for pid in tqdm(range(1, 43)):
        try:
            resp = Resp.from_wav(f"../data/Respiration/P{pid:03d}.wav")
            resp.remove_baseline(method='savgol')
            resp.find_cycles(include_holds=True)

            inh_durs = [i.duration() for i in resp.inhalations]
            exh_durs = [i.duration() for i in resp.exhalations]

            if inh_durs and exh_durs:
                inh_dur_means.append(np.mean(inh_durs))
                inh_dur_stds.append(np.std(inh_durs))
                exh_dur_means.append(np.mean(exh_durs))
                exh_dur_stds.append(np.std(exh_durs))

        except Exception as e:
            print(f"Skipping P{pid:03d}: {e}")

    inh_dur_means = np.array(inh_dur_means)
    inh_dur_stds = np.array(inh_dur_stds)
    exh_dur_means = np.array(exh_dur_means)
    exh_dur_stds = np.array(exh_dur_stds)

    np.save(cache_files["inh_dur_means"], inh_dur_means)
    np.save(cache_files["inh_dur_stds"], inh_dur_stds)
    np.save(cache_files["exh_dur_means"], exh_dur_means)
    np.save(cache_files["exh_dur_stds"], exh_dur_stds)

# --------------------
# Plotting
# --------------------
labels = ['Inhalation', 'Exhalation', 'Inhalation Variability', 'Exhalation Variability']
x = np.arange(len(labels))
width = 0.6

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Violin plot for durations (ax1)
vp1 = ax1.violinplot(
    [inh_dur_means, exh_dur_means],
    positions=[0, 1],
    widths=width,
    showmedians=False,
    showextrema=True
)
ax1.vlines(0, np.percentile(inh_dur_means, 25), np.percentile(inh_dur_means, 75), color='black', linestyle='-', lw=10)
ax1.vlines(1, np.percentile(exh_dur_means, 25), np.percentile(exh_dur_means, 75), color='black', linestyle='-', lw=10)
# Violin plot for variability (ax2)
vp2 = ax2.violinplot(
    [inh_dur_stds, exh_dur_stds],
    positions=[2, 3],
    widths=width,
    showmedians=False,
    showextrema=True
)
ax2.vlines(2, np.percentile(inh_dur_stds, 25), np.percentile(inh_dur_stds, 75), color='black', linestyle='-', lw=10)
ax2.vlines(3, np.percentile(exh_dur_stds, 25), np.percentile(exh_dur_stds, 75), color='black', linestyle='-', lw=10)
# Color formatting
for b in vp1['bodies']:
    b.set_facecolor('gray')
    b.set_edgecolor('black')
    b.set_alpha(0.4)
for b in vp2['bodies']:
    b.set_facecolor('gray')
    b.set_edgecolor('black')
    b.set_alpha(0.4)
vp1['cmins'].set_color('black')     # min whisker color
vp1['cmaxes'].set_color('black')    # max whisker color
vp1['cbars'].set_color('black')     # vertical line between min/max
vp2['cmins'].set_color('black')     # min whisker color
vp2['cmaxes'].set_color('black')    # max whisker color
vp2['cbars'].set_color('black')     # vertical line between min/max

# Axes labels
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Duration (s)")
ax2.set_ylabel("Standard Deviation (s)", rotation=-90, labelpad=20)
ax1.set_xlabel("Respiration Timing Metrics")

# Grid & spines
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.grid(axis='x', visible=False)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
