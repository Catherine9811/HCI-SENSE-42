import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import mne
import addcopyfighandler

# Plot style
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

# Constants
event_id = 5
tmin, tmax = -0.1, 0.5  # ERP window
xmin, xmax = -0.1, 0.5
use_cache = True
cache_folder = "data"
cache_file = os.path.join(cache_folder, f"erp_event{event_id}_{tmin}s-{tmax}s_all_participants.npy")
times_file = os.path.join(cache_folder, f"erp_event{event_id}_times_all_participants.npy")
events_folder = os.path.join("data", "events")

if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

participants = range(1, 43) # [2, 6, 8, 19, 25, 26, 29, 30, 31, 33, 34, 39, 40, 42]
# participants = participants[7:8]
if not use_cache or not os.path.exists(cache_file) or not os.path.exists(times_file):
    erp_list = []
    times = None

    for participant_id in tqdm(participants):
        # Load EEG
        raw = mne.io.read_raw_eeglab(
            rf"D:\HCI PROCESSED DATA\CleanedEEGLab\P{participant_id:03d}.set",
            preload=True, eog=["REF"]
        )
        raw.set_eeg_reference("average")
        # raw.resample(200, npad="auto")  # Downsample to 200 Hz

        # Pick standard EEG channels
        raw.pick_channels([
            # "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3",
            # "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
            # "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz"
            # "PO3", "PO4"
            # "Cz"
            "Fz"
        ])

        sfreq = raw.info['sfreq']

        # Load events from .txt file
        event_file = os.path.join(events_folder, f"P{participant_id:03d}.txt")
        custom_events = []

        with open(event_file, 'r') as f:
            lines = f.readlines()[1:]  # skip header
            for line in lines:
                _, evt_type, onset_sec = line.strip().split('\t')
                evt_type = int(evt_type)
                onset_sample = int(float(onset_sec) * sfreq)
                custom_events.append([onset_sample, 0, evt_type])

        custom_events = np.array(custom_events)

        # Make Epochs
        epochs = mne.Epochs(
            raw,
            custom_events,
            event_id={str(event_id): event_id},
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0),
            detrend=1,
            preload=True
        )

        # Compute mean ERP (average across trials, then across channels)
        evoked = epochs[str(event_id)].average()
        times = evoked.times
        erp_data = evoked.data.mean(axis=0)  # mean over channels
        erp_list.append(erp_data)

        del raw

    # Save results
    erp_stack = np.stack(erp_list)
    np.save(cache_file, erp_stack)
    np.save(times_file, times)
else:
    erp_stack = np.load(cache_file)
    times = np.load(times_file)

# Plot ERP: black = mean, gray = SEM
mean_erp = erp_stack.mean(axis=0)
sem_erp = erp_stack.std(axis=0) / np.sqrt(erp_stack.shape[0] - 1)


# Define a smoothing function (moving average)
def smooth(data, window_len=20):
    window = np.ones(window_len) / window_len
    return np.convolve(data, window, mode='same')


# Apply smoothing
window_len = 10  # adjust depending on your sampling rate
mean_erp = smooth(mean_erp, window_len)
sem_erp = smooth(sem_erp, window_len)

plt.figure(figsize=(8, 5))
plt.plot(times * 1000, mean_erp * 1e6, color='black', label='mean ERP')
plt.fill_between(times * 1000, mean_erp * 1e6 - sem_erp * 1e6, mean_erp * 1e6 + sem_erp * 1e6, color='gray', alpha=0.4)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
# plt.title(f"Event-Related Potential")
plt.grid(True, linestyle='--', alpha=0.5)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
plt.xlim(xmin * 1000, xmax * 1000)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
