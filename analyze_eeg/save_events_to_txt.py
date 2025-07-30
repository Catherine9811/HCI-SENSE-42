import mne
import os
import numpy as np

EEG_ORIGINAL_SAMPLING_FREQUENCY = 1024    # Hz

# Directory containing the BDF files
bdf_dir = r"D:\HCI PROCESSED DATA\EEGRaw"
save_dir = os.path.join("data", "events")
info_file = os.path.join("data", "default-info.fif.gz")

# Participant IDs to process (e.g., 1 to 20)
participant_ids = range(1, 43)

for participant_id in participant_ids:
    bdf_file = os.path.join(bdf_dir, f"P{participant_id:03d}.bdf")

    if not os.path.exists(bdf_file):
        print(f"File not found: {bdf_file}")
        continue

    print(f"Processing {bdf_file}")

    # Load raw BDF file
    raw = mne.io.read_raw_bdf(bdf_file, preload=True)

    if participant_id == 5:
        raw_extra = mne.io.read_raw_bdf(os.path.join(bdf_dir, f"P{participant_id:03d} 02.bdf"), preload=True)
        raw = mne.concatenate_raws([raw, raw_extra])

    if participant_id == 1:
        mne.io.write_info(info_file, raw.info)

    # Extract events from the stimulus channel
    events = mne.find_events(raw, stim_channel='Status', initial_event=True, shortest_event=1)

    # Prepare output filename
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_file = os.path.join(save_dir, f"P{participant_id:03d}.txt")

    # Save as tab-separated file: index (row), type (event_id), onset (sample)
    with open(output_file, 'w') as f:
        f.write("Index\tType\tOnset\n")
        for i, (frame, _, event_id) in enumerate(events):
            f.write(f"{i}\t{event_id}\t{frame/EEG_ORIGINAL_SAMPLING_FREQUENCY}\n")

    print(f"Saved events to {output_file}")
