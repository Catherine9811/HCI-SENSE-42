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
    DEFINITION.CALIB_BEGIN: "calibration_start.started",
    DEFINITION.CALIB_END: "calibration_end.started",
    DEFINITION.WINDOW_CLOSE_BEGIN: "window_close.started",
    DEFINITION.MAIL_HOMESCREEN_BEGIN: "mail_homescreen.started",
    DEFINITION.MAIL_NOTIFICATION_BEGIN: "mail_notification.started",
    DEFINITION.MAIL_CONTENT_BEGIN: "mail_content.started",
    DEFINITION.FILE_MANAGER_HOMESCREEN_BEGIN: "file_manager_homescreen.started",
    DEFINITION.FILE_MANAGER_DRAGGING_BEGIN: "file_manager_dragging.started",
    DEFINITION.FILE_MANAGER_OPENING_BEGIN: "file_manager_opening.started",
    DEFINITION.TRASH_BIN_HOMESCREEN_BEGIN: "trash_bin_homescreen.started",
    DEFINITION.TRASH_BIN_SELECT_BEGIN: "trash_bin_select.started",
    DEFINITION.TRASH_BIN_CONFIRM_BEGIN: "trash_bin_confirm.started",
    DEFINITION.NOTES_HOMESCREEN_BEGIN: "notes_homescreen.started",
    DEFINITION.NOTES_REPEAT_BEGIN: "notes_repeat.started",
    DEFINITION.BROWSER_HOMESCREEN_BEGIN: "browser_homescreen.started",
    DEFINITION.BROWSER_NAVIGATION_BEGIN: "browser_navigation.started",
    DEFINITION.BROWSER_CONTENT_BEGIN: "browser_content.started"
}
EEG_PAUSE_BLOCK = "pause_on"

missing = 0
total = 0

for psydat_file in tqdm(psydat_files[1:]):
    participant_id = int(psydat_file.split("_")[0])
    parser = DataParser(os.path.join("..", "data", psydat_file))

    # Define the file path
    bdf_file = rf"D:\HCI PROCESSED DATA\EEGRaw\P{participant_id:03d}.bdf"

    # Load the raw BDF file
    raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose='WARNING')
    if participant_id == 5:
        raw_extra = mne.io.read_raw_bdf(rf"D:\HCI PROCESSED DATA\EEGRaw\P{participant_id:03d} 02.bdf", preload=True)
        raw = mne.concatenate_raws([raw, raw_extra])

    # Extract events from stimulus channel
    events = mne.find_events(raw, stim_channel='Status', initial_event=True, shortest_event=1, verbose='WARNING')

    # Create event_id mapping
    event_id = {str(event): event for event in set(events[:, 2]) if event < 255}

    for signal_definition, task_definition in EEG_SIGNAL_DEFINITION.items():
        task_name = task_definition.split(".")[0]
        matched_entries = parser[task_definition]
        matched_events = events[:, 0][events[:, 2] == signal_definition]
        total += len(matched_entries)
        missing += abs(len(matched_entries) - len(matched_events))
        if len(matched_events) != len(matched_entries):
            print(f"P{participant_id:03d} Unmatched {task_name} ({signal_definition}): "
                  f"Events ({len(matched_events)}), Entries ({len(matched_entries)})")

print("Missing", missing)
print("Total", total)
print("Percentage", f"{missing/total:.3f} %")