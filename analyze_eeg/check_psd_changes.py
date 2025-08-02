import os

import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import addcopyfighandler

# Use Arial for all text
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14               # base font size
mpl.rcParams['axes.titlesize'] = 16          # title font size
mpl.rcParams['axes.labelsize'] = 14          # axis label font size
mpl.rcParams['xtick.labelsize'] = 12         # x-tick label size
mpl.rcParams['ytick.labelsize'] = 12         # y-tick label size
mpl.rcParams['legend.fontsize'] = 12         # legend font size

# Load EEG data
first_list = []
last_list = []
freqs = []
fmin, fmax = 0.5, 20
pmin, pmax = -7.5, 20

use_cache = True
cache_folder = "data"
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

if not use_cache or \
        not os.path.exists(os.path.join(cache_folder, "psd_first_30min.npy")) or \
        not os.path.exists(os.path.join(cache_folder, "psd_last_30min.npy")):
    for participant_id in tqdm(range(1, 43)):
        if participant_id not in [2, 6, 8, 19, 25, 26, 29, 30, 31, 33, 34, 39, 40, 42]:
            continue
        raw = mne.io.read_raw_eeglab(rf"D:\HCI PROCESSED DATA\AutomagicCleanedEEGLab\P{participant_id:03d}.set", preload=True, eog=["REF"])
        raw = raw.pick_channels(ch_names=[
            "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3",
            "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
            "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz",
            "Cz"
        ])

        raw = raw.set_eeg_reference("average")

        raw = raw.pick_channels(ch_names=[
            "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3",
            "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
            "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz",
            "Cz"
        ])

        # Calculate time durations
        total_duration = raw.times[-1]
        segment_duration = 30 * 60  # 30 minutes

        # First and last 30-minute segments
        first_segment = raw.copy().crop(tmin=0, tmax=segment_duration)
        last_segment = raw.copy().crop(tmin=total_duration - segment_duration, tmax=total_duration)

        # Compute PSDs
        psds_first = first_segment.compute_psd(fmin=fmin, fmax=fmax, n_fft=2048)
        psds_last = last_segment.compute_psd(fmin=fmin, fmax=fmax, n_fft=2048)

        # Extract data
        freqs = psds_first.freqs
        psds_first_db = 10 * np.log10(psds_first.get_data()) + 120
        psds_last_db = 10 * np.log10(psds_last.get_data()) + 120

        first_list.append(psds_first_db.mean(axis=0))
        last_list.append(psds_last_db.mean(axis=0))

        del raw

    # Mean and Std across channels
    first_stack = np.stack(first_list)
    mean_first = first_stack.mean(axis=0)
    std_first = first_stack.std(axis=0) / np.sqrt(first_stack.shape[0] - 1)
    last_stack = np.stack(last_list)
    mean_last = last_stack.mean(axis=0)
    std_last = last_stack.std(axis=0) / np.sqrt(last_stack.shape[0] - 1)

    np.save(os.path.join(cache_folder, "psd_first_30min.npy"), first_stack)
    np.save(os.path.join(cache_folder, "psd_last_30min.npy"), last_stack)
else:
    first_stack = np.load(os.path.join(cache_folder, "psd_first_30min.npy"))
    mean_first = first_stack.mean(axis=0)
    std_first = first_stack.std(axis=0) / np.sqrt(first_stack.shape[0] - 1)
    last_stack = np.load(os.path.join(cache_folder, "psd_last_30min.npy"))
    mean_last = last_stack.mean(axis=0)
    std_last = last_stack.std(axis=0) / np.sqrt(last_stack.shape[0] - 1)
    freqs = np.linspace(fmin, fmax, len(mean_last))

# Define frequency range: 8.0–14.0 Hz
fmin_band = 8.0
fmax_band = 14.0
freq_mask = (freqs >= fmin_band) & (freqs <= fmax_band)

# Average power within 8–14 Hz for each participant
power_first_avg = first_stack[:, freq_mask].mean(axis=1)
power_last_avg = last_stack[:, freq_mask].mean(axis=1)

# Paired t-test on averaged power
t_stat, p_value = ttest_rel(power_first_avg, power_last_avg)
# Report p-value
print(f"Paired t-test at {fmin_band}-{fmax_band} Hz: t = {t_stat:.3f}, p = {p_value:.4f}")


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
    ax.plot([x1, x1, x2, x2, x1], [y1, max(y1, y2) + h, max(y1, y2) + h, y2, y1], color='black', linewidth=1.5)
    # Add stars text
    ax.text((x1 + x2) * 0.5, max(y1, y2) + h + text_offset, stars, font="Arial",
            ha='center', va='bottom', color='black', fontsize=12)


# Plot
plt.figure(figsize=(8, 5))

# First 30 minutes
plt.plot(freqs, mean_first, label='first 30 min', color='tab:blue')
plt.fill_between(freqs, mean_first - std_first, mean_first + std_first, color='tab:blue', alpha=0.3)

# Last 30 minutes
plt.plot(freqs, mean_last, label='last 30 min', color='tab:orange')
plt.fill_between(freqs, mean_last - std_last, mean_last + std_last, color='tab:orange', alpha=0.3)

# Aesthetics
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
# plt.title('Power Spectral Density: First vs Last 30 Minutes')
plt.xlim(fmin, fmax)
plt.ylim(pmin, pmax)
plt.grid(True, linestyle='--', alpha=0.5)
custom_legend = [
    Patch(facecolor='tab:blue', alpha=1.0),
    Patch(facecolor='tab:orange', alpha=1.0)
]
# legend = plt.legend(custom_legend, ["first 30 min", "last 30 min"], loc='upper right', frameon=True, borderpad=0.6)
# legend.get_frame().set_edgecolor('black')  # set solid edge color
# legend.get_frame().set_linewidth(1.0)      # optional: control thickness
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

y1 = max(first_stack[:, np.argmin(np.abs(freqs - fmin_band))].mean(), last_stack[:, np.argmin(np.abs(freqs - fmin_band))].mean())
y2 = max(first_stack[:, np.argmin(np.abs(freqs - fmax_band))].mean(), last_stack[:, np.argmin(np.abs(freqs - fmax_band))].mean())

add_significance_annotation(ax=ax, x1=fmin_band, x2=fmax_band, y1=-6, y2=-6, h=8, p_val=p_value, text_offset=0.4)

plt.tight_layout()

# Show or save
plt.show()
