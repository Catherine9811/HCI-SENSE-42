import os
import mne
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import addcopyfighandler

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
    "rr_medians": os.path.join(cache_folder, "rr_medians_all.npy"),
    "rr_stds": os.path.join(cache_folder, "rr_stds_all.npy"),
    "participant_ids": os.path.join(cache_folder, "participant_ids_all.npy")
}

# Load from cache if available
if use_cache and all(os.path.exists(f) for f in cache_files.values()):
    rr_medians = np.load(cache_files["rr_medians"])
    rr_stds = np.load(cache_files["rr_stds"])
    participant_ids = np.load(cache_files["participant_ids"])
else:
    rr_medians = []
    rr_stds = []
    participant_ids = []

    for pid in tqdm(range(2, 43)):
        try:
            fif_file = rf"../data/ECG/P{pid:03d}.fif"
            if not os.path.exists(fif_file):
                continue

            raw_ecg = mne.io.read_raw_fif(fif_file, preload=True, verbose="WARNING")
            ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw_ecg, ch_name='ECG2', tstart=120.0, verbose="WARNING")

            sfreq = raw_ecg.info['sfreq']
            event_times = ecg_events[:, 0] / sfreq
            rr_intervals = np.diff(event_times)

            min_bpm = 50
            max_bpm = 220
            valid = np.logical_and(rr_intervals >= 60.0 / max_bpm, rr_intervals <= 60.0 / min_bpm)
            rr_intervals = rr_intervals[valid] * 1000  # Convert to ms

            if len(rr_intervals) > 0:
                rr_medians.append(np.median(rr_intervals))
                rr_stds.append(np.std(rr_intervals))
                participant_ids.append(f"P{pid:03d}")
        except Exception as e:
            print(f"Error processing P{pid:03d}: {e}")

    rr_medians = np.array(rr_medians)
    rr_stds = np.array(rr_stds)
    participant_ids = np.array(participant_ids)

    # Save to cache
    np.save(cache_files["rr_medians"], rr_medians)
    np.save(cache_files["rr_stds"], rr_stds)
    np.save(cache_files["participant_ids"], participant_ids)

# Plotting
labels = ['R-R Interval', 'Heart Rate Variability']
width = 0.3
rr_pos = 0
hrv_pos = 0.5
x = [rr_pos, hrv_pos]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Violin for RR intervals (left y-axis)
vp1 = ax1.violinplot([rr_medians],
                     positions=[rr_pos],
                     widths=width, showmeans=False, showmedians=False, showextrema=True)
ax1.vlines(rr_pos, np.percentile(rr_medians, 25), np.percentile(rr_medians, 75), color='black', linestyle='-', lw=10)
ax1.vlines(rr_pos, np.min(rr_medians), np.max(rr_medians), color='black', linestyle='-', lw=1)
for b in vp1['bodies']:
    b.set_facecolor('gray')
    b.set_edgecolor('black')
    b.set_alpha(0.4)
if 'cmedians' in vp1:
    vp1['cmedians'].set_color('black')
    vp1['cmedians'].set_linewidth(2)
vp1['cmins'].set_color('black')     # min whisker color
vp1['cmaxes'].set_color('black')    # max whisker color
vp1['cbars'].set_color('black')     # vertical line between min/max

rr_stds = rr_stds[rr_stds < 160]
# Violin for HRV (right y-axis)
vp2 = ax2.violinplot([rr_stds],
                     positions=[hrv_pos],
                     widths=width, showmeans=False, showmedians=False, showextrema=True)
ax2.vlines(hrv_pos, np.percentile(rr_stds, 25), np.percentile(rr_stds, 75), color='black', linestyle='-', lw=10)
ax2.vlines(hrv_pos, np.min(rr_stds), np.max(rr_stds), color='black', linestyle='-', lw=1)
for b in vp2['bodies']:
    b.set_facecolor('gray')
    b.set_edgecolor('black')
    b.set_alpha(0.4)
if 'cmedians' in vp2:
    vp2['cmedians'].set_color('black')
    vp2['cmedians'].set_linewidth(2)
vp2['cmins'].set_color('black')     # min whisker color
vp2['cmaxes'].set_color('black')    # max whisker color
vp2['cbars'].set_color('black')     # vertical line between min/max
# Set x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# Y-axis labels
ax1.set_ylabel("Duration (ms)")
ax2.set_ylabel("Standard Deviation (ms)", rotation=-90, labelpad=20)
ax1.set_xlabel("Cardiac Timing Metrics")

# Aesthetics
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.grid(axis='x', visible=False)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
