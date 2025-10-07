import os
import numpy as np
from tqdm import tqdm
import mne

# === Settings ===
participants = range(1, 43)
events_folder = os.path.join("data", "events")
saving_psd_dir = os.path.join("data", "psd_welch_event7to9")
fmin, fmax = 0.0, 100.0  # frequency range

if not os.path.exists(saving_psd_dir):
    os.makedirs(saving_psd_dir)


# === Main loop ===
for participant_id in (bar := tqdm(participants)):
    bar.set_description(f"P{participant_id:03d}")

    # --- Load EEG ---
    eeglab_file = rf"D:\HCI PROCESSED DATA\CleanedEEGLab\P{participant_id:03d}.set"
    raw = mne.io.read_raw_eeglab(eeglab_file, preload=True, verbose="WARNING")
    raw = raw.set_eeg_reference("average")
    sfreq = raw.info["sfreq"]

    # --- Load events ---
    event_file = os.path.join(events_folder, f"P{participant_id:03d}.txt")
    custom_events = []
    with open(event_file, "r") as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            _, evt_type, onset_sec = line.strip().split("\t")
            evt_type = int(evt_type)
            onset_sample = int(float(onset_sec) * sfreq)
            custom_events.append([onset_sample, 0, evt_type])
    custom_events = np.array(custom_events)

    # --- Find all event 7 and 9 times ---
    evt7_samples = custom_events[custom_events[:, 2] == 7][:, 0]
    evt9_samples = custom_events[custom_events[:, 2] == 9][:, 0]

    # Loop over all pairs
    pair_idx = 0
    for s7 in evt7_samples:
        # find the first event9 *after* this event7
        s9_after = evt9_samples[evt9_samples > s7]
        if len(s9_after) == 0:
            continue  # no following event9
        s9 = s9_after[0]

        tmin = s7 / sfreq
        tmax = s9 / sfreq
        if tmax <= tmin:
            continue

        # --- Crop raw to this window ---
        raw_crop = raw.copy().crop(tmin=tmin, tmax=tmax)

        # --- Compute PSD across the whole interval ---
        psds, freqs = raw_crop.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            tmin=0,
            tmax=None,   # entire cropped duration
        ).get_data(return_freqs=True)

        # --- Save ---
        pair_idx += 1
        out_path = os.path.join(
            saving_psd_dir,
            f"P{participant_id:03d}-psd-event7to9-{pair_idx:02d}.npz"
        )
        np.savez_compressed(
            out_path,
            channels=raw.ch_names,
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            psds=psds,
            freqs=freqs,
        )
