import numpy as np
import pickle
import os
import functools
import pandas as pd
import jellyfish
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser
from data_definition import psydat_files


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_ASPECT_RATIO = 1920 / 1080


def filter_time_series(times, values):
    return times, values
    # times = np.array(times)
    # values = np.array(values)
    # condition = (times < 2 * 60 * 60)
    # print(f"Time filtered {np.sum(condition)} items.")
    # return times[condition], values[condition]


def filter_nan_indices(processed):
    """Remove all indices where any value in the dictionary contains NaN."""
    # Find indices where any column contains NaN
    nan_indices = {i for values in processed.values() for i, v in enumerate(values) if np.isnan(v)}

    # Keep only the indices that are NOT in nan_indices
    processed = {key: [v for i, v in enumerate(values) if i not in nan_indices] for key, values in processed.items()}

    print(f"{len(set(nan_indices))} filtered.")

    return processed


def to_title_case(s):
    return s.replace("_", " ").title()


def scale_mouse_measurement_to_screen_height(measure):
    if isinstance(measure, tuple):
        return measure[0] * SCREEN_ASPECT_RATIO, measure[1]
    measure[0] *= SCREEN_ASPECT_RATIO
    return measure


def scale_width_measurement_to_screen_height(width):
    return width * SCREEN_ASPECT_RATIO


class ComputerUsageComfortExtractor:
    name = "comfort"

    def __init__(self, csv_path):
        self.enrollment = pd.read_csv(csv_path)
        self.id_column = "Participant ID"
        self.uh_column = "How comfortable are you with using computers?"
        # Extract the relevant columns
        data = self.enrollment[[self.id_column, self.uh_column]].dropna()

        # Create a dictionary: {participant_id (int): str}
        self.uh_dict = {
            int(row[self.id_column]): str(row[self.uh_column])
            for _, row in data.iterrows()
        }

    def process(self, participant_id: int):
        return self.uh_dict.get(participant_id, "")


class UsingOperatingSystemMode:
    def get_entry_condition(self, parser, item):
        stylizer = parser["operating_system_style"]
        style_mapping = {
            entry["trials.thisN"]: entry["operating_system_style"]
            for entry in stylizer if 'trials.thisN' in entry
        }
        return style_mapping[item[f"trials.thisN"]]

    def get_condition(self, parser, task):
        stylizer = parser["operating_system_style"]
        style_mapping = {
            entry["trials.thisN"]: entry["operating_system_style"]
            for entry in stylizer if 'trials.thisN' in entry
        }
        return [style_mapping[entry[f"trials.thisN"]] for entry in task]


class MouseDoubleClickDistanceExtractor(UsingOperatingSystemMode):
    name = "mouse_double_click_distance"

    def process(self, parser):
        task_key = 'stimuli_reaction_mouse_movement.started'
        task = parser[task_key]
        # Extracting values for plotting
        x_values = self.get_condition(parser, task)
        y_values = []
        for entry in task:
            target_location = entry["stimuli_reaction_stimuli_location"]
            last_mouse_location_x = entry[task_key.replace(".started", ".x")][-1]
            last_mouse_location_y = entry[task_key.replace(".started", ".y")][-1]
            distance = np.linalg.norm(scale_mouse_measurement_to_screen_height(np.array([
                target_location[0] - last_mouse_location_x,
                target_location[1] - last_mouse_location_y
            ])))
            y_values.append(distance)
        return x_values, y_values


class MouseDragDistanceExtractor(UsingOperatingSystemMode):
    name = "mouse_drag_distance"

    def process(self, parser):
        task_key = 'stimuli_dragging_mouse.started'
        task = parser[task_key]
        # Extracting values for plotting
        x_values = self.get_condition(parser, task)
        y_values = [np.linalg.norm(scale_mouse_measurement_to_screen_height(entry["last_clicked_offset"])) for entry in task]
        return x_values, y_values


class MouseDropDistanceExtractor(UsingOperatingSystemMode):
    name = "mouse_drop_distance"

    def process(self, parser):
        task_key = 'stimuli_dragging_mouse.started'
        task = parser[task_key]
        # Extracting values for plotting
        x_values = self.get_condition(parser, task)
        y_values = []
        for entry in task:
            last_mouse_pos = np.array(
                [entry["last_clicked_offset"][0] + entry["file_dragging.stimuli_dragging_mouse.x"][-1],
                 entry["last_clicked_offset"][1] + entry["file_dragging.stimuli_dragging_mouse.y"][-1]])
            target_mouse_pos = np.array(entry["target_location"])
            y_values.append(np.linalg.norm(scale_mouse_measurement_to_screen_height(last_mouse_pos - target_mouse_pos)))
        return x_values, y_values


class MouseTaskbarNavigationEfficiencyExtractor(UsingOperatingSystemMode):
    name = "mouse_taskbar_navigation_efficiency"

    @staticmethod
    def effective_movement_ratio(x, y):
        if len(x) < 2 or len(y) < 2:
            return 0  # Not enough points to calculate movement

        # Compute straight-line distance (Euclidean distance between first and last point)
        straight_line_distance = np.hypot(scale_width_measurement_to_screen_height(x[-1] - x[0]), y[-1] - y[0])

        # Compute total trajectory distance (sum of Euclidean distances between consecutive points)
        total_distance = sum(np.hypot(scale_width_measurement_to_screen_height(np.diff(x)), np.diff(y)))

        # Avoid division by zero
        return straight_line_distance / total_distance if total_distance > 0 else 0

    def process(self, parser):
        x_values = []
        y_values = []
        for task_key in [
            'file_manager_mouse_homescreen.started',
            'trash_bin_mouse_homescreen.started',
            'notes_mouse_homescreen.started',
            'browser_mouse_homescreen.started'
        ]:
            typing_task = parser[task_key]
            # Extracting values for plotting
            x_values.extend(self.get_condition(parser, typing_task))

            for entry in typing_task:
                # Get right keys for entry
                candidates = [key for key in entry.keys()
                              if key.endswith(f".{task_key.replace('started', 'time')}") and len(entry[key]) > 0]
                assert len(candidates) > 0, f"{candidates}"
                key = candidates[-1]
                x = entry[key.replace(".time", ".x")]
                y = entry[key.replace(".time", ".y")]
                y_values.append(MouseTaskbarNavigationEfficiencyExtractor.effective_movement_ratio(x, y))
        return x_values, y_values


class MouseToolbarNavigationEfficiencyExtractor(UsingOperatingSystemMode):
    name = "mouse_toolbar_navigation_efficiency"

    @staticmethod
    def effective_movement_ratio(x, y):
        if len(x) < 2 or len(y) < 2:
            return 0  # Not enough points to calculate movement

        # Compute straight-line distance (Euclidean distance between first and last point)
        straight_line_distance = np.hypot(scale_width_measurement_to_screen_height(x[-1] - x[0]), y[-1] - y[0])

        # Compute total trajectory distance (sum of Euclidean distances between consecutive points)
        total_distance = sum(np.hypot(scale_width_measurement_to_screen_height(np.diff(x)), np.diff(y)))

        # Avoid division by zero
        return straight_line_distance / total_distance if total_distance > 0 else 0

    def process(self, parser):
        x_values = []
        y_values = []
        for task_key in [
            'window_close_mouse.started'
        ]:
            typing_task = parser[task_key]
            typing_task = [entry for entry in typing_task if entry['window_close_target_name'] == 'Close']

            # Extracting values for plotting
            x_values.extend(self.get_condition(parser, typing_task))

            for entry in typing_task:
                # Get right keys for entry
                candidates = [key for key in entry.keys()
                              if key.endswith(f".{task_key.replace('started', 'time')}") and len(entry[key]) > 0]
                assert len(candidates) > 0, f"{candidates}"
                key = candidates[-1]
                x = entry[key.replace(".time", ".x")]
                y = entry[key.replace(".time", ".y")]
                y_values.append(MouseTaskbarNavigationEfficiencyExtractor.effective_movement_ratio(x, y))
        return x_values, y_values


class MouseSelectionCoverageExtractor(UsingOperatingSystemMode):
    name = "mouse_selection_coverage"

    def process(self, parser):
        task_key = 'trash_bin_select_mouse.started'
        task = parser[task_key]

        # Extracting values for plotting
        x_values = self.get_condition(parser, task)
        y_values = []
        for entry in task:
            # Collect location of all folders
            area_start = [1.0, 1.0]
            area_end = [-1.0, -1.0]
            selection_start = entry['selection_start']
            selection_end = entry['selection_end']
            for key_name in entry:
                if key_name.startswith('trash_folder_'):
                    area_start[0] = min(area_start[0], entry[key_name][0])
                    area_start[1] = min(area_start[1], entry[key_name][1])
                    area_end[0] = max(area_end[0], entry[key_name][0])
                    area_end[1] = max(area_end[1], entry[key_name][1])
            optimal_area_size = abs((area_end[1] - area_start[1]) * scale_width_measurement_to_screen_height(area_end[0] - area_start[0]))
            actual_area_size = abs((selection_end[1] - selection_start[1]) * scale_width_measurement_to_screen_height(selection_end[0] - selection_start[0]))
            y_values.append(optimal_area_size / actual_area_size)
        return x_values, y_values


class MouseFolderNavigationSpeedExtractor(UsingOperatingSystemMode):
    name = "mouse_folder_navigation_speed"
    
    @staticmethod
    def total_movements(x, y):
        if len(x) < 2 or len(y) < 2:
            return 0  # Not enough points to calculate movement

        # Compute total trajectory distance (sum of Euclidean distances between consecutive points)
        total_distance = sum(np.hypot(scale_width_measurement_to_screen_height(np.diff(x)), np.diff(y)))

        # Avoid division by zero
        return total_distance

    def process(self, parser):
        task_key = 'stimuli_reaction_mouse_movement.started'
        typing_task = parser[task_key]
        # Extracting values for plotting
        x_values = self.get_condition(parser, typing_task)

        y_values = []

        for entry in typing_task:
            # Get right keys for entry
            candidates = [key for key in entry.keys()
                          if key.endswith(f".{task_key.replace('started', 'time')}") and len(entry[key]) > 0]
            assert len(candidates) > 0, f"{candidates}"
            key = candidates[-1]
            x = entry[key.replace(".time", ".x")]
            y = entry[key.replace(".time", ".y")]
            y_values.append(MouseFolderNavigationSpeedExtractor.total_movements(x, y) / entry[key][-1])
        return x_values, y_values


class MouseToolbarNavigationSpeedExtractor(UsingOperatingSystemMode):
    name = "mouse_toolbar_navigation_speed"

    @staticmethod
    def total_movements(x, y):
        if len(x) < 2 or len(y) < 2:
            return 0  # Not enough points to calculate movement

        # Compute total trajectory distance (sum of Euclidean distances between consecutive points)
        total_distance = sum(np.hypot(scale_width_measurement_to_screen_height(np.diff(x)), np.diff(y)))

        # Avoid division by zero
        return total_distance

    def process(self, parser):
        task_key = 'window_close_mouse.started'
        typing_task = parser[task_key]
        typing_task = [entry for entry in typing_task if entry['window_close_target_name'] == 'Close']

        # Extracting values for plotting
        x_values = self.get_condition(parser, typing_task)

        y_values = []

        for entry in typing_task:
            # Get right keys for entry
            candidates = [key for key in entry.keys()
                          if key.endswith(f".{task_key.replace('started', 'time')}") and len(entry[key]) > 0]
            assert len(candidates) > 0, f"{candidates}"
            key = candidates[-1]
            x = entry[key.replace(".time", ".x")]
            y = entry[key.replace(".time", ".y")]
            y_values.append(MouseToolbarNavigationSpeedExtractor.total_movements(x, y) / entry[key][-1])
        return x_values, y_values


class TaskCompletionDurationBase(UsingOperatingSystemMode):
    name = "task_completion_duration_base"
    task_key = None

    def process(self, parser):
        typing_task = parser[self.task_key]

        # Extracting values for plotting
        x_values = self.get_condition(parser, typing_task)
        y_values = [entry[f"{self.task_key}.stopped"] - entry[f"{self.task_key}.started"] for entry in typing_task]
        return x_values, y_values


class MouseOpenFileManagerDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_open_file_manager_duration"
    task_key = 'file_manager_homescreen'


class MouseOpenTrashBinDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_open_trash_bin_duration"
    task_key = 'trash_bin_homescreen'


class MouseOpenNotesDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_open_notes_duration"
    task_key = 'notes_homescreen'


class MouseOpenBrowserDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_open_browser_duration"
    task_key = 'browser_homescreen'


class MouseConfirmDialogDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_confirm_dialog_duration"
    task_key = 'trash_bin_confirm'


class MouseNotificationDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_notification_duration"
    task_key = 'mail_notification'


class MouseOpenFolderDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_open_folder_duration"
    task_key = 'file_manager_opening'


class MouseDragFolderDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_drag_folder_duration"
    task_key = 'file_manager_dragging'


class MouseCloseWindowDurationExtractor(UsingOperatingSystemMode):
    name = "mouse_close_window_duration"
    task_key = 'window_close'

    def process(self, parser):
        typing_task = parser[self.task_key]
        typing_task = [entry for entry in typing_task if entry['window_close_target_name'] == 'Close']
        # Extracting values for plotting
        x_values = self.get_condition(parser, typing_task)
        y_values = [entry[f"{self.task_key}.stopped"] - entry[f"{self.task_key}.started"] for entry in typing_task]
        return x_values, y_values


class MouseGroupedSelectionDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_grouped_selection_duration"
    task_key = 'trash_bin_select'


class KeyboardShadowTypingDurationExtractor(TaskCompletionDurationBase):
    name = "keyboard_shadow_typing_duration"
    task_key = 'mail_content'


class KeyboardSideBySideTypingDurationExtractor(TaskCompletionDurationBase):
    name = "keyboard_side_by_side_typing_duration"
    task_key = 'notes_repeat'


class KeyboardShadowTypingErrorExtractor(UsingOperatingSystemMode):
    name = "keyboard_shadow_typing_error"

    def process(self, parser):
        task_key = 'mail_content'
        task_keyboard = 'mail.mail_content_user_key_release'
        typing_task = parser[task_key]
        # Extracting values for plotting
        x_values = self.get_condition(parser, typing_task)
        y_values = []
        for entry in typing_task:
            y_values.append(entry[f"{task_keyboard}.keys"].count("backspace"))
        return x_values, y_values


class KeyboardSideBySideTypingErrorExtractor(UsingOperatingSystemMode):
    name = "keyboard_side_by_side_typing_error"

    def process(self, parser):
        task_key = 'notes_repeat'
        task_keyboard = 'notes.notes_repeat_keyboard'
        typing_task = parser[task_key]
        # Extracting values for plotting
        x_values = self.get_condition(parser, typing_task)
        y_values = []
        for entry in typing_task:
            y_values.append(entry[f"{task_keyboard}.keys"].count("backspace"))
        return x_values, y_values


class KeyboardTypingSpeedExtractor(UsingOperatingSystemMode):
    name = "keyboard_typing_speed"

    @staticmethod
    def count_effective_keys(keys):
        return len(keys) - 2 * keys.count('backspace')

    def process(self, parser):
        x_values = []
        y_values = []
        for task_name, task_key, task_prefix, task_keyboard in [
            ('Shadow Typing', 'mail_content', 'single_note', 'mail.mail_content_user_key_release'),
            ('Side-by-side Typing', 'notes_repeat', 'notes', 'notes.notes_repeat_keyboard')
        ]:
            typing_task = parser[task_key]
            # Extracting values for plotting
            x_values.extend(self.get_condition(parser, typing_task))
            y_values.extend([
                KeyboardTypingSpeedExtractor.count_effective_keys(entry[f"{task_keyboard}.keys"]) / (
                            entry[f"{task_key}.stopped"] - entry[f"{task_key}.started"])
                for entry in typing_task])
        return x_values, y_values


class KeyboardShadowTypingEfficiencyExtractor(UsingOperatingSystemMode):
    name = "keyboard_shadow_typing_efficiency"

    def process(self, parser):
        task_key = 'mail_content'
        task_keyboard = 'mail.mail_content_user_key_release'
        task_prefix = 'single_note'
        typing_task = parser[task_key]
        # Extracting values for plotting
        x_values = self.get_condition(parser, typing_task)
        y_values = []
        for entry in typing_task:
            typing_speed = KeyboardTypingSpeedExtractor.count_effective_keys(entry[f"{task_keyboard}.keys"]) / (
                            entry[f"{task_key}.stopped"] - entry[f"{task_key}.started"])
            typing_error = jellyfish.jaro_similarity(entry[f"{task_prefix}_repeat_source"],
                                                     entry[f"{task_prefix}_repeat_target"])
            y_values.append(
                typing_speed * typing_error
            )
        return x_values, y_values


class KeyboardSideBySideTypingEfficiencyExtractor(UsingOperatingSystemMode):
    name = "keyboard_side_by_side_typing_efficiency"

    def process(self, parser):
        task_key = 'notes_repeat'
        task_keyboard = 'notes.notes_repeat_keyboard'
        task_prefix = 'notes'
        typing_task = parser[task_key]
        # Extracting values for plotting
        x_values = self.get_condition(parser, typing_task)
        y_values = []
        for entry in typing_task:
            typing_speed = KeyboardTypingSpeedExtractor.count_effective_keys(entry[f"{task_keyboard}.keys"]) / (
                            entry[f"{task_key}.stopped"] - entry[f"{task_key}.started"])
            typing_error = jellyfish.jaro_similarity(entry[f"{task_prefix}_repeat_source"],
                                                     entry[f"{task_prefix}_repeat_target"])
            y_values.append(
                typing_speed * typing_error
            )
        return x_values, y_values


class KeyboardSpaceKeyTypingDurationExtractor(UsingOperatingSystemMode):
    name = "keyboard_space_key_typing_duration"

    def process(self, parser):
        x_values = []
        y_values = []
        for task_name, task_key, task_prefix, task_keyboard in [
            ('Shadow Typing', 'mail_content', 'single_note', 'mail.mail_content_user_key_release'),
            ('Side-by-side Typing', 'notes_repeat', 'notes', 'notes.notes_repeat_keyboard')
        ]:
            typing_task = parser[task_key]
            # Extracting values for plotting
            for entry in typing_task:
                recorded_start_time = entry[f"{task_key}.started"]
                recorded_keys = entry[f"{task_keyboard}.keys"]
                recorded_rt = entry[f"{task_keyboard}.rt"]
                recorded_duration = entry[f"{task_keyboard}.duration"]
                for key, rt, duration in zip(recorded_keys, recorded_rt, recorded_duration):
                    if key == "space":
                        x_values.append(self.get_entry_condition(parser, entry))
                        y_values.append(rt)
        return x_values, y_values


class KeyboardSpaceKeyPressedDurationExtractor(UsingOperatingSystemMode):
    name = "keyboard_space_key_pressed_duration"

    def process(self, parser):
        x_values = []
        y_values = []
        for task_name, task_key, task_prefix, task_keyboard in [
            ('Shadow Typing', 'mail_content', 'single_note', 'mail.mail_content_user_key_release'),
            ('Side-by-side Typing', 'notes_repeat', 'notes', 'notes.notes_repeat_keyboard')
        ]:
            typing_task = parser[task_key]
            # Extracting values for plotting
            for entry in typing_task:
                recorded_start_time = entry[f"{task_key}.started"]
                recorded_keys = entry[f"{task_keyboard}.keys"]
                recorded_rt = entry[f"{task_keyboard}.rt"]
                recorded_duration = entry[f"{task_keyboard}.duration"]
                for key, rt, duration in zip(recorded_keys, recorded_rt, recorded_duration):
                    if key == "space":
                        x_values.append(self.get_entry_condition(parser, entry))
                        y_values.append(duration)
        return x_values, y_values


class KeyboardPressedDurationExtractor(UsingOperatingSystemMode):
    name = "keyboard_pressed_duration"

    def process(self, parser):
        x_values = []
        y_values = []
        for task_name, task_key, task_prefix, task_keyboard in [
            ('Shadow Typing', 'mail_content', 'single_note', 'mail.mail_content_user_key_release'),
            ('Side-by-side Typing', 'notes_repeat', 'notes', 'notes.notes_repeat_keyboard')
        ]:
            typing_task = parser[task_key]
            # Extracting values for plotting
            for entry in typing_task:
                recorded_start_time = entry[f"{task_key}.started"]
                recorded_rt = entry[f"{task_keyboard}.rt"]
                recorded_duration = entry[f"{task_keyboard}.duration"]
                for rt, duration in zip(recorded_rt, recorded_duration):
                    x_values.append(self.get_entry_condition(parser, entry))
                    y_values.append(duration)
        return x_values, y_values


if __name__ == '__main__':
    variable_definitions = [
        MouseDoubleClickDistanceExtractor(),
        MouseDragDistanceExtractor(),
        MouseDropDistanceExtractor(),
        MouseTaskbarNavigationEfficiencyExtractor(),
        MouseToolbarNavigationEfficiencyExtractor(),
        MouseSelectionCoverageExtractor(),
        MouseFolderNavigationSpeedExtractor(),
        MouseToolbarNavigationSpeedExtractor(),
        MouseConfirmDialogDurationExtractor(),
        MouseNotificationDurationExtractor(),
        MouseOpenFolderDurationExtractor(),
        MouseDragFolderDurationExtractor(),
        MouseCloseWindowDurationExtractor(),
        MouseGroupedSelectionDurationExtractor(),
        MouseOpenNotesDurationExtractor(),
        MouseOpenBrowserDurationExtractor(),
        MouseOpenFileManagerDurationExtractor(),
        MouseOpenTrashBinDurationExtractor(),
        KeyboardShadowTypingDurationExtractor(),
        KeyboardSideBySideTypingDurationExtractor(),
        KeyboardShadowTypingErrorExtractor(),
        KeyboardSideBySideTypingErrorExtractor(),
        KeyboardTypingSpeedExtractor(),
        KeyboardSpaceKeyPressedDurationExtractor(),
        KeyboardSpaceKeyTypingDurationExtractor(),
        KeyboardPressedDurationExtractor(),
        KeyboardShadowTypingEfficiencyExtractor(),
        KeyboardSideBySideTypingEfficiencyExtractor()
    ]

    enrollment = os.path.join("..", "..", "data", "participant_enrollment.csv")
    condition_definitions = ComputerUsageComfortExtractor(enrollment)

    processed = {}

    for psydat_file in tqdm(psydat_files):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        for extractor in variable_definitions:
            predictor_name = extractor.name
            predictor_os, predictor_values = extractor.process(parser)
            assert len(predictor_os) == len(predictor_values), "Size doesn't match!"
            predictor_associated_values = []
            predictor_associated_vars = []
            predictor_os = np.array(predictor_os)
            predictor_values = np.array(predictor_values)

            if "name" not in processed:
                processed["name"] = []
            processed["name"].extend([predictor_name] * len(predictor_values))
            if "comfort" not in processed:
                processed["comfort"] = []
            processed["comfort"].extend([condition_definitions.process(participant_id)] * len(predictor_values))
            if "os" not in processed:
                processed["os"] = []
            processed["os"].extend(predictor_os)
            if "value" not in processed:
                processed["value"] = []
            processed["value"].extend(predictor_values)
            if "participant" not in processed:
                processed["participant"] = []
            processed["participant"].extend([participant_id] * len(predictor_values))

    # Convert processed dictionary to DataFrame
    df = pd.DataFrame(processed)

    save_directory = os.path.join("processed_data", "behavioural")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save to CSV file
    df.to_csv(os.path.join(save_directory, f"{len(psydat_files)}-{condition_definitions.name}.csv"), index=False)

    print("Processed data saved.")

