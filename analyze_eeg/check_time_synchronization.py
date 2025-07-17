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

times_to_plot = []
errors_to_plot = []
colors_to_plot = []

for psydat_file in tqdm(psydat_files[7:8]):
    participant_id = int(psydat_file.split("_")[0])
    parser = DataParser(os.path.join("..", "data", psydat_file))

    # Paused blocks
    paused_blocks = parser[EEG_PAUSE_BLOCK]

    if len(paused_blocks) > 0:
        print(f"P{participant_id:03d} PAUSED {len(paused_blocks)} TIMES!!!")

    # Define the file path
    bdf_file = rf"../data/EEG/P{participant_id:03d}.bdf"

    # Load the raw BDF file
    raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose='WARNING')

    # Extract events from stimulus channel
    events = mne.find_events(raw, stim_channel='Status', initial_event=True, shortest_event=1, verbose='WARNING')

    # Create event_id mapping
    event_id = {str(event): event for event in set(events[:, 2]) if event < 255}

    for signal_definition, task_definition in EEG_SIGNAL_DEFINITION.items():
        task_name = task_definition.split(".")[0]
        matched_entries = parser[task_definition]
        matched_events = events[:, 0][events[:, 2] == signal_definition]
        # if len(matched_events) != len(matched_entries):
        #     print(f"P{participant_id:03d} Unmatched {task_name} ({signal_definition}): "
        #           f"Events ({len(matched_events)}), Entries ({len(matched_entries)})")
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

        # adjusted_event_times = matched_events / EEG_SAMPLING_FREQUENCY

        if len(adjusted_event_times) == 0:
            print(f"P{participant_id:03d} {task_name} ({signal_definition}) EVENT_NOT_FOUND "
                  f"Events ({len(matched_events)}), Entries ({len(matched_entries)})")
            continue

        # Store mapping and errors
        mapping = {}
        errors = []

        for entry in matched_entries:
            timestamp = entry[task_definition]
            # Find the closest adjusted EEG time
            closest_idx = np.argmin(np.abs(np.array(adjusted_event_times) - timestamp))
            closest_time = adjusted_event_times[closest_idx]
            mapping[timestamp] = closest_time
            errors.append(abs(timestamp - closest_time))
            times_to_plot.append(timestamp)
            errors_to_plot.append(timestamp - closest_time)
            colors_to_plot.append(participant_id)
        print(f"P{participant_id:03d} {task_name} ({signal_definition}) Error {np.max(errors):.05f}: "
              f"Events ({len(matched_events)}), Entries ({len(matched_entries)})")

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Scatter plot: each point is an error
ax.scatter(times_to_plot, errors_to_plot, c=colors_to_plot, alpha=0.2, edgecolor='white')

# Titles and labels
ax.set_title("Timestamp Matching Errors between EEG Events and Behavioural Data", fontsize=16)
ax.set_xlabel("Behavioural Timestamp", fontsize=14)
ax.set_ylabel("Absolute Error (seconds)", fontsize=14)

# Grid for readability
ax.grid(True, linestyle='--', alpha=0.5)

# Improve look: remove top/right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tighter layout
plt.tight_layout()
plt.show()