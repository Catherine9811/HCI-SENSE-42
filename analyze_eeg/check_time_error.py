import os
import mne
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from serial_connection_definition import SerialConnectionDefinition as DEFINITION
from data_parser import DataParser
from data_definition import psydat_files

EEG_SAMPLING_FREQUENCY = 1024    # Hz
EEG_SIGNAL_DEFINITION = {
    # DEFINITION.CALIB_BEGIN: "calibration_start.started",
    # DEFINITION.CALIB_END: "calibration_end.started",
    # DEFINITION.WINDOW_CLOSE_BEGIN: "window_close.started",
    # DEFINITION.MAIL_HOMESCREEN_BEGIN: "mail_homescreen.started",
    # DEFINITION.MAIL_NOTIFICATION_BEGIN: "mail_notification.started",
    # DEFINITION.MAIL_CONTENT_BEGIN: "mail_content.started",
    # DEFINITION.FILE_MANAGER_HOMESCREEN_BEGIN: "file_manager_homescreen.started",
    DEFINITION.FILE_MANAGER_DRAGGING_BEGIN: "file_manager_dragging.started",
    # DEFINITION.FILE_MANAGER_OPENING_BEGIN: "file_manager_opening.started",
    # DEFINITION.TRASH_BIN_HOMESCREEN_BEGIN: "trash_bin_homescreen.started",
    # DEFINITION.TRASH_BIN_SELECT_BEGIN: "trash_bin_select.started",
    # DEFINITION.TRASH_BIN_CONFIRM_BEGIN: "trash_bin_confirm.started",
    # DEFINITION.NOTES_HOMESCREEN_BEGIN: "notes_homescreen.started",
    # DEFINITION.NOTES_REPEAT_BEGIN: "notes_repeat.started",
    # DEFINITION.BROWSER_HOMESCREEN_BEGIN: "browser_homescreen.started",
    # DEFINITION.BROWSER_NAVIGATION_BEGIN: "browser_navigation.started",
    # DEFINITION.BROWSER_CONTENT_BEGIN: "browser_content.started"
}
EEG_PAUSE_BLOCK = "pause_on"


use_cache = True
cache_folder = "data"
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

if use_cache and \
        os.path.exists(os.path.join(cache_folder, "behavioural_times.npy")) and \
        os.path.exists(os.path.join(cache_folder, "eeg_times.npy")):
    behavioural_times = np.load(os.path.join(cache_folder, "behavioural_times.npy"))
    eeg_times = np.load(os.path.join(cache_folder, "eeg_times.npy"))
else:
    behavioural_times = []
    eeg_times = []

    for psydat_file in tqdm(psydat_files[1:]):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "data", psydat_file))

        # Paused blocks
        paused_blocks = parser[EEG_PAUSE_BLOCK]

        if len(paused_blocks) > 0:
            print(f"P{participant_id:03d} PAUSED {len(paused_blocks)} TIMES!!!")

        # Define the file path
        bdf_file = rf"D:\HCI PROCESSED DATA\EEGRaw\P{participant_id:03d}.bdf"

        # Load the raw BDF file
        raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose='WARNING')
        if participant_id == 5:
            raw_extra = mne.io.read_raw_bdf(rf"D:\HCI PROCESSED DATA\EEGRaw\P{participant_id:03d} 02.bdf",
                                            preload=True)
            raw = mne.concatenate_raws([raw, raw_extra])

        # Extract events from stimulus channel
        events = mne.find_events(raw, stim_channel='Status', initial_event=True, shortest_event=1, verbose='WARNING')

        # Create event_id mapping
        event_id = {str(event): event for event in set(events[:, 2]) if event < 255}

        for signal_definition, task_definition in EEG_SIGNAL_DEFINITION.items():
            task_name = task_definition.split(".")[0]
            matched_entries = parser[task_definition]
            matched_events = events[:, 0][events[:, 2] == signal_definition]
            # Convert EEG sample indices to seconds
            adjusted_event_times = []
            for event_sample in matched_events:
                time_in_sec = event_sample / EEG_SAMPLING_FREQUENCY
                total_pause_duration = sum(
                    block[f"{EEG_PAUSE_BLOCK}.stopped"] - block[f"{EEG_PAUSE_BLOCK}.started"]
                    for block in paused_blocks
                    if time_in_sec > block[f"{EEG_PAUSE_BLOCK}.started"]
                )
                adjusted_time = time_in_sec + total_pause_duration
                adjusted_event_times.append(adjusted_time)

            if len(adjusted_event_times) == 0:
                print(f"P{participant_id:03d} {task_name} ({signal_definition}) EVENT_NOT_FOUND "
                      f"Events ({len(matched_events)}), Entries ({len(matched_entries)})")
                continue

            for entry in matched_entries:
                timestamp = entry[task_definition]
                # Find the closest adjusted EEG time
                closest_idx = np.argmin(np.abs(np.array(adjusted_event_times) - timestamp))
                closest_time = adjusted_event_times[closest_idx]
                behavioural_times.append(timestamp)
                eeg_times.append(closest_time)
    behavioural_times = np.array(behavioural_times)
    eeg_times = np.array(eeg_times)
    np.save(os.path.join(cache_folder, "behavioural_times.npy"), behavioural_times)
    np.save(os.path.join(cache_folder, "eeg_times.npy"), eeg_times)

# diff = np.abs(behavioural_times - eeg_times)
# threshold = 5
# behavioural_times = behavioural_times[diff < threshold]
# eeg_times = eeg_times[diff < threshold]

behavioural_times = np.diff(behavioural_times)
eeg_times = np.diff(eeg_times)

timing_plot = behavioural_times - eeg_times
timing_diff = np.abs(behavioural_times - eeg_times)

limitation = 0.05
timing_plot = timing_plot[timing_diff < limitation]
timing_diff = timing_diff[timing_diff < limitation]

print("Timing Accuracy", "mean:", np.mean(timing_plot), "std:", np.std(timing_plot))

plt.hist(timing_plot, bins=100)
plt.xlabel("Time Difference (s)")
plt.ylabel("Count")
plt.show()
