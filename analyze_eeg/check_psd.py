import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import addcopyfighandler

# Use Arial for all text
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

# Settings
fmin, fmax = 0.5, 20
pmin, pmax = -7.5, 20
use_cache = True
cache_folder = "data"
cache_file = os.path.join(cache_folder, "psd_all_participants.npy")

if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

# Load or compute PSDs across all time
if not use_cache or not os.path.exists(cache_file):
    all_list = []
    freqs = None

    for participant_id in tqdm(range(1, 43)):
        # if participant_id not in [2, 6, 8, 19, 25, 26, 29, 30, 31, 33, 34, 39, 40, 42]:
        #     continue

        raw = mne.io.read_raw_eeglab(
            rf"D:\HCI PROCESSED DATA\CleanedEEGLab\P{participant_id:03d}.set",
            preload=True, eog=["REF"]
        )

        raw = raw.pick_channels([
            "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3",
            "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
            "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz"
        ])

        raw.set_eeg_reference("average")

        # Compute PSD for full recording
        psds = raw.compute_psd(fmin=fmin, fmax=fmax, n_fft=2048)
        freqs = psds.freqs
        psds_db = 10 * np.log10(psds.get_data()) + 120
        all_list.append(psds_db.mean(axis=0))  # mean across channels

        del raw

    # Stack and save
    all_stack = np.stack(all_list)
    np.save(cache_file, all_stack)
else:
    all_stack = np.load(cache_file)
    freqs = np.linspace(fmin, fmax, all_stack.shape[1])

# Compute mean and standard error
mean_all = all_stack.mean(axis=0)
std_all = all_stack.std(axis=0) / np.sqrt(all_stack.shape[0] - 1)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(freqs, mean_all, color='black', label='mean')
plt.fill_between(freqs, mean_all - std_all, mean_all + std_all, color='gray', alpha=0.4)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.xlim(fmin, fmax)
plt.ylim(pmin, pmax)
plt.grid(True, linestyle='--', alpha=0.5)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
