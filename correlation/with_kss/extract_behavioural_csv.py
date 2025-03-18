import numpy as np
import pickle
import os
import pandas as pd
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

SMOOTH_WINDOW = 5 * 60


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


class PerformanceAnswerExtractor:
    name = "performance"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "performance:_how_successful_were_you_in_accomplishing_what_you_were_asked_to_do_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class TemporalDemandAnswerExtractor:
    name = "temporal_demand"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "temporal_demand:_how_hurried_or_rushed_was_the_pace_of_the_task_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class AttentivenessAnswerExtractor:
    name = "attentiveness"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "attentiveness:_how_focused_were_you_on_performing_the_task_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class SleepinessAnswerExtractor:
    name = "sleepiness"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "sleepiness:_how_sleepy_are_you_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class MouseDoubleClickDistanceExtractor:
    name = "mouse_double_click_distance"

    def process(self, parser):
        task_key = 'stimuli_reaction_mouse_movement.started'
        task = parser[task_key]
        # Extracting values for plotting
        x_values = [entry[task_key] for entry in task]
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


class MouseDragDistanceExtractor:
    name = "mouse_drag_distance"

    def process(self, parser):
        task_key = 'stimuli_dragging_mouse.started'
        task = parser[task_key]
        # Extracting values for plotting
        x_values = [entry[task_key] for entry in task]
        y_values = [np.linalg.norm(scale_mouse_measurement_to_screen_height(entry["last_clicked_offset"])) for entry in task]
        return x_values, y_values


class MouseDropDistanceExtractor:
    name = "mouse_drop_distance"

    def process(self, parser):
        task_key = 'stimuli_dragging_mouse.started'
        task = parser[task_key]
        # Extracting values for plotting
        x_values = [entry[task_key] for entry in task]
        y_values = []
        for entry in task:
            last_mouse_pos = np.array(
                [entry["last_clicked_offset"][0] + entry["file_dragging.stimuli_dragging_mouse.x"][-1],
                 entry["last_clicked_offset"][1] + entry["file_dragging.stimuli_dragging_mouse.y"][-1]])
            target_mouse_pos = np.array(entry["target_location"])
            y_values.append(np.linalg.norm(scale_mouse_measurement_to_screen_height(last_mouse_pos - target_mouse_pos)))
        return x_values, y_values


class MouseTaskbarNavigationEfficiencyExtractor:
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
            x_values.extend([entry[f"{task_key}"] for entry in typing_task])

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


class MouseToolbarNavigationEfficiencyExtractor:
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
            # Extracting values for plotting
            x_values.extend([entry[f"{task_key}"] for entry in typing_task])

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


class MouseSelectionCoverageExtractor:
    name = "mouse_selection_coverage"

    def process(self, parser):
        task_key = 'trash_bin_select_mouse.started'
        task = parser[task_key]

        # Extracting values for plotting
        x_values = [entry[task_key] for entry in task]
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


class MouseFolderNavigationSpeedExtractor:
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
        x_values = [entry[f"{task_key}"] for entry in typing_task]

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


class MouseToolbarNavigationSpeedExtractor:
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
        # Extracting values for plotting
        x_values = [entry[f"{task_key}"] for entry in typing_task]

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


class TaskCompletionDurationBase:
    name = "task_completion_duration_base"
    task_key = None

    def process(self, parser):
        typing_task = parser[self.task_key]

        # Extracting values for plotting
        x_values = [entry[f"{self.task_key}.started"] for entry in typing_task]
        y_values = [entry[f"{self.task_key}.stopped"] - entry[f"{self.task_key}.started"] for entry in typing_task]
        return x_values, y_values


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


class MouseCloseWindowDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_close_window_duration"
    task_key = 'window_close'


class MouseGroupedSelectionDurationExtractor(TaskCompletionDurationBase):
    name = "mouse_grouped_selection_duration"
    task_key = 'trash_bin_select'


class KeyboardShadowTypingDurationExtractor(TaskCompletionDurationBase):
    name = "keyboard_shadow_typing_duration"
    task_key = 'mail_content'


class KeyboardSideBySideTypingDurationExtractor(TaskCompletionDurationBase):
    name = "keyboard_side_by_side_typing_duration"
    task_key = 'notes_repeat'


class KeyboardShadowTypingErrorExtractor:
    name = "keyboard_shadow_typing_error"

    def process(self, parser):
        task_key = 'mail_content'
        task_keyboard = 'mail.mail_content_user_key_release'
        typing_task = parser[task_key]
        # Extracting values for plotting
        x_values = [entry[f"{task_key}.started"] for entry in typing_task]
        y_values = []
        for entry in typing_task:
            y_values.append(entry[f"{task_keyboard}.keys"].count("backspace"))
        return x_values, y_values


class KeyboardSideBySideTypingErrorExtractor:
    name = "keyboard_side_by_side_typing_error"

    def process(self, parser):
        task_key = 'notes_repeat'
        task_keyboard = 'notes.notes_repeat_keyboard'
        typing_task = parser[task_key]
        # Extracting values for plotting
        x_values = [entry[f"{task_key}.started"] for entry in typing_task]
        y_values = []
        for entry in typing_task:
            y_values.append(entry[f"{task_keyboard}.keys"].count("backspace"))
        return x_values, y_values


class KeyboardTypingSpeedExtractor:
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
            x_values.extend([entry[f"{task_key}.started"] for entry in typing_task])
            y_values.extend([
                KeyboardTypingSpeedExtractor.count_effective_keys(entry[f"{task_keyboard}.keys"]) / (
                            entry[f"{task_key}.stopped"] - entry[f"{task_key}.started"])
                for entry in typing_task])
        return x_values, y_values


class KeyboardSpaceKeyTypingDurationExtractor:
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
                        x_values.append(recorded_start_time + rt)
                        y_values.append(rt)
        return x_values, y_values


class KeyboardSpaceKeyPressedDurationExtractor:
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
                        x_values.append(recorded_start_time + rt)
                        y_values.append(duration)
        return x_values, y_values


class KeyboardPressedDurationExtractor:
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
                    x_values.append(recorded_start_time + rt)
                    y_values.append(duration)
        return x_values, y_values


if __name__ == '__main__':
    predictor_definitions = [
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
        KeyboardShadowTypingDurationExtractor(),
        KeyboardSideBySideTypingDurationExtractor(),
        KeyboardShadowTypingErrorExtractor(),
        KeyboardSideBySideTypingErrorExtractor(),
        KeyboardTypingSpeedExtractor(),
        KeyboardSpaceKeyPressedDurationExtractor(),
        KeyboardSpaceKeyTypingDurationExtractor(),
        KeyboardPressedDurationExtractor(),
    ]

    outcome_definition = TemporalDemandAnswerExtractor()

    processed = {}

    for psydat_file in tqdm(psydat_files):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        outcome_times, outcome_values = outcome_definition.process(parser)
        outcome_times, outcome_values = filter_time_series(outcome_times, outcome_values)
        for extractor in predictor_definitions:
            predictor_name = extractor.name
            predictor_times, predictor_values = extractor.process(parser)
            predictor_associated_values = []
            predictor_associated_vars = []
            predictor_times = np.array(predictor_times)
            predictor_values = np.array(predictor_values)
            for outcome_time in outcome_times:
                associated_values = predictor_values[(predictor_times > outcome_time - SMOOTH_WINDOW) & (predictor_times <= outcome_time)]
                predictor_associated_values.append(np.mean(associated_values))
                predictor_associated_vars.append(np.std(associated_values))
            if f"{predictor_name}_mean" not in processed:
                processed[f"{predictor_name}_mean"] = []
            processed[f"{predictor_name}_mean"].extend(predictor_associated_values)
            if f"{predictor_name}_var" not in processed:
                processed[f"{predictor_name}_var"] = []
            processed[f"{predictor_name}_var"].extend(predictor_associated_vars)
        if "time" not in processed:
            processed["time"] = []
        processed["time"].extend(outcome_times)
        if outcome_definition.name not in processed:
            processed[outcome_definition.name] = []
        processed[outcome_definition.name].extend(outcome_values)
        if "participant" not in processed:
            processed["participant"] = []
        processed["participant"].extend([participant_id] * len(outcome_values))

    processed = filter_nan_indices(processed)

    # Convert processed dictionary to DataFrame
    df = pd.DataFrame(processed)

    save_directory = os.path.join("processed_data", "behavioural")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save to CSV file
    df.to_csv(os.path.join(save_directory, f"{len(psydat_files)}-{outcome_definition.name}.csv"), index=False)

    print("Processed data saved.")

