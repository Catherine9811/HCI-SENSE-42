import os
import mne
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
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
    "rr_first_means": os.path.join(cache_folder, "rr_first_means.npy"),
    "rr_last_means": os.path.join(cache_folder, "rr_last_means.npy"),
    "rr_first_stds": os.path.join(cache_folder, "rr_first_stds.npy"),
    "rr_last_stds": os.path.join(cache_folder, "rr_last_stds.npy"),
    "participant_ids": os.path.join(cache_folder, "participant_ids.npy")
}
# Load from cache if available
if use_cache and all(os.path.exists(f) for f in cache_files.values()):
    rr_first_means = np.load(cache_files["rr_first_means"])
    rr_last_means = np.load(cache_files["rr_last_means"])
    rr_first_stds = np.load(cache_files["rr_first_stds"])
    rr_last_stds = np.load(cache_files["rr_last_stds"])
    participant_ids = np.load(cache_files["participant_ids"])
else:
    # Initialize lists
    rr_first_means = []
    rr_last_means = []
    rr_first_stds = []
    rr_last_stds = []
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
            rr_time = (event_times[:-1] + event_times[1:]) / 2

            min_bpm = 50
            max_bpm = 220
            valid = np.logical_and(rr_intervals >= 60.0 / max_bpm, rr_intervals <= 60.0 / min_bpm)
            rr_intervals = rr_intervals[valid]
            rr_time = rr_time[valid]

            duration = raw_ecg.times[-1]
            start_first = 0
            end_first = 30 * 60
            start_last = duration - 30 * 60
            end_last = duration

            # First 30 min
            idx_first = np.where((rr_time >= start_first) & (rr_time < end_first))[0]
            rr_first = rr_intervals[idx_first] * 1000  # convert to ms

            # Last 30 min
            idx_last = np.where((rr_time >= start_last) & (rr_time < end_last))[0]
            rr_last = rr_intervals[idx_last] * 1000  # convert to ms

            # Only if both segments have data
            if len(rr_first) > 0 and len(rr_last) > 0:
                rr_first_means.append(np.median(rr_first))
                rr_first_stds.append(np.std(rr_first))
                rr_last_means.append(np.median(rr_last))
                rr_last_stds.append(np.std(rr_last))
                participant_ids.append(f"P{pid:03d}")
        except Exception as e:
            print(f"Error processing P{pid:03d}: {e}")

    rr_first_means = np.array(rr_first_means)
    rr_last_means = np.array(rr_last_means)
    rr_first_stds = np.array(rr_first_stds)
    rr_last_stds = np.array(rr_last_stds)
    participant_ids = np.array(participant_ids)

    # Save to cache
    np.save(cache_files["rr_first_means"], rr_first_means)
    np.save(cache_files["rr_last_means"], rr_last_means)
    np.save(cache_files["rr_first_stds"], rr_first_stds)
    np.save(cache_files["rr_last_stds"], rr_last_stds)
    np.save(cache_files["participant_ids"], participant_ids)


n = len(rr_first_means)

# Compute group means and SEM
mean_rr_first = rr_first_means.mean()
mean_rr_last = rr_last_means.mean()
sem_rr_first = rr_first_means.std(ddof=1) / np.sqrt(n)
sem_rr_last = rr_last_means.std(ddof=1) / np.sqrt(n)

mean_hrv_first = rr_first_stds.mean()
mean_hrv_last = rr_last_stds.mean()
sem_hrv_first = rr_first_stds.std(ddof=1) / np.sqrt(n)
sem_hrv_last = rr_last_stds.std(ddof=1) / np.sqrt(n)

from scipy.stats import ttest_rel

print("Paired t-test for R-R Interval (First vs Last):")
t_rr, p_rr = ttest_rel(rr_first_means, rr_last_means)
print(f"t={t_rr:.3f}, p={p_rr:.9f}")

print("Paired t-test for HRV (First vs Last):")
t_hrv, p_hrv = ttest_rel(rr_first_stds, rr_last_stds)
print(f"t={t_hrv:.3f}, p={p_hrv:.9f}")


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


# Group positions
labels = ['R-R Interval', 'Heart Rate Variability']
x = np.arange(len(labels))

# Violin positions
width = 0.2
rr_pos = 0
hrv_pos = 1

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Draw a subtle line (light gray, semi-transparent) for each paired observation
# for a, b in zip(rr_first_means, rr_last_means):
#     ax1.plot([rr_pos - width, rr_pos + width], [a, b], color='black', alpha=0.05, linewidth=5, zorder=1)
# R-R Interval on ax1 (left axis)
vp1 = ax1.violinplot([rr_first_means, rr_last_means],
                     positions=[rr_pos - width, rr_pos + width],
                     widths=width, showmeans=False, showmedians=True, showextrema=False)
for b in vp1['bodies'][0:1]:
    b.set_facecolor('tab:blue')
    b.set_edgecolor('black')
    b.set_alpha(0.7)
    b.set_zorder(3)
for b in vp1['bodies'][1:2]:
    b.set_facecolor('tab:orange')
    b.set_edgecolor('black')
    b.set_alpha(0.7)
    b.set_zorder(3)
vp1['cmedians'].set_color('black')
vp1['cmedians'].set_alpha(0.7)
vp1['cmedians'].set_zorder(4)
# HRV on ax2 (right axis)
vp2 = ax2.violinplot([rr_first_stds, rr_last_stds],
                     positions=[hrv_pos - width, hrv_pos + width],
                     widths=width, showmeans=False, showmedians=True, showextrema=False)
for b in vp2['bodies'][0:1]:
    b.set_facecolor('tab:blue')
    b.set_edgecolor('black')
    b.set_alpha(0.7)
for b in vp2['bodies'][1:2]:
    b.set_facecolor('tab:orange')
    b.set_edgecolor('black')
    b.set_alpha(0.7)
vp2['cmedians'].set_color('black')
vp2['cmedians'].set_alpha(0.7)

# Set x-axis
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# Y-axis labels
ax1.set_ylabel("Duration (ms)")
ax2.set_ylabel("Variance (ms)", rotation=-90, labelpad=20)
ax1.set_xlabel("Cardiac Timing Metrics")

# Aesthetics
# ax1.set_title("R-R Interval and HRV (Violin Plot) Across Participants")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.grid(axis='x', visible=False)  # hide y-axis grid
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Custom legend
handles = [
    Patch(facecolor='tab:blue', edgecolor='black', label='first 30 min'),
    Patch(facecolor='tab:orange', edgecolor='black', label='last 30 min')
]
# legend = ax1.legend(handles=handles, loc='upper center', frameon=True, borderpad=0.6)
# legend.get_frame().set_edgecolor('black')  # set solid edge color
# legend.get_frame().set_linewidth(1.0)      # optional: control thickness

# Define where to place annotations (y values)
# You might want to set y above the max of your plotted data + some margin
max_rr = max(np.max(rr_first_means), np.max(rr_last_means))
min_rr = min(np.min(rr_first_means), np.min(rr_last_means))
# Add annotation for RR (x positions for your first group)
add_significance_annotation(ax1, x1=rr_pos - width, x2=rr_pos + width, y1=np.max(rr_first_means)*1.05, y2=np.max(rr_last_means)*1.05, h=max_rr*0.05, p_val=p_rr, text_offset=0.02*(max_rr - min_rr))

# Assuming ax2 is your secondary y-axis, and your HRV groups are at x=1 and x=2
# Find max y value of HRV data plotted on ax2 (adjust this to your data)
max_hrv_val = max(np.max(rr_first_stds), np.max(rr_last_stds))
min_hrv_val = min(np.min(rr_first_stds), np.min(rr_last_stds))
# Draw the annotation on ax2
add_significance_annotation(ax2, x1=hrv_pos - width, x2=hrv_pos + width, y1=np.max(rr_first_stds)*1.05, y2=np.max(rr_last_stds)*1.05, h=max_hrv_val*0.05, p_val=p_hrv, text_offset=0.02*(max_hrv_val - min_hrv_val))


plt.tight_layout()
plt.show()
