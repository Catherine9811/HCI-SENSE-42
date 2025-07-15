import numpy as np
import pickle
import os
import functools
import pandas as pd
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


class HandednessExtractor:
    name = "keyboard_handedness"

    def __init__(self, csv_path):
        self.enrollment = pd.read_csv(csv_path)
        self.id_column = "Participant ID"
        self.uh_column = "Handedness"
        # Extract the relevant columns
        data = self.enrollment[[self.id_column, self.uh_column]].dropna()

        # Create a dictionary: {participant_id (int): str}
        self.uh_dict = {
            int(row[self.id_column]): str(row[self.uh_column])
            for _, row in data.iterrows()
        }

    def process(self, participant_id: int):
        return self.uh_dict.get(participant_id, "")


class KeyboardTypingDurationExtractor:
    name = "keyboard_typing_duration"
    readable_name = "Hesitation Time before Pressing a Keyboard Key (s)"
    reduce_mode = "average"
    group_by = "handedness"
    max_clipping = 15

    def process(self, parser):
        x_values = []
        y_values = []
        for task_name, task_key, task_prefix, task_keyboard in [
            ('Shadow Typing', 'mail_content', 'single_note', 'mail.mail_content_user_key_release'),
            ('Side-by-side Typing', 'notes_repeat', 'notes', 'notes.notes_repeat_keyboard'),
            ('Browser URL Typing', 'browser_navigation', 'browser', 'browser.browser_navigation_user_key_release')
        ]:
            typing_task = parser[task_key]
            # Extracting values for plotting
            for entry in typing_task:
                recorded_keys = entry[f"{task_keyboard}.keys"]
                recorded_rt = entry[f"{task_keyboard}.rt"]
                recorded_duration = entry[f"{task_keyboard}.duration"]
                for key, rt, duration in zip(recorded_keys, recorded_rt, recorded_duration):
                    x_values.append(key)
                    y_values.append(rt)
        return x_values, y_values


class KeyboardPressedDurationExtractor:
    name = "keyboard_pressed_duration"
    readable_name = "Pressed Duration of Keyboard Keys (s)"
    reduce_mode = "average"
    max_clipping = 5
    # group_by = "handedness"
    color_max_clipping = 2.0

    def process(self, parser):
        x_values = []
        y_values = []
        for task_name, task_key, task_prefix, task_keyboard in [
            ('Shadow Typing', 'mail_content', 'single_note', 'mail.mail_content_user_key_release'),
            ('Side-by-side Typing', 'notes_repeat', 'notes', 'notes.notes_repeat_keyboard'),
            ('Browser URL Typing', 'browser_navigation', 'browser', 'browser.browser_navigation_user_key_release')
        ]:
            typing_task = parser[task_key]
            # Extracting values for plotting
            for entry in typing_task:
                recorded_keys = entry[f"{task_keyboard}.keys"]
                recorded_rt = entry[f"{task_keyboard}.rt"]
                recorded_duration = entry[f"{task_keyboard}.duration"]
                for key, rt, duration in zip(recorded_keys, recorded_rt, recorded_duration):
                    x_values.append(key)
                    y_values.append(duration)
        return x_values, y_values


class KeyboardKeyCountExtractor:
    name = "keyboard_key_count"
    readable_name = "Logarithm Count of Pressed Keyboard Keys"
    reduce_mode = "sum"
    max_clipping = 999

    def process(self, parser):
        x_values = []
        y_values = []
        for task_name, task_key, task_prefix, task_keyboard in [
            ('Shadow Typing', 'mail_content', 'single_note', 'mail.mail_content_user_key_release'),
            ('Side-by-side Typing', 'notes_repeat', 'notes', 'notes.notes_repeat_keyboard'),
            ('Browser URL Typing', 'browser_navigation', 'browser', 'browser.browser_navigation_user_key_release')
        ]:
            typing_task = parser[task_key]
            # Extracting values for plotting
            for entry in typing_task:
                recorded_keys = entry[f"{task_keyboard}.keys"]
                for key in recorded_keys:
                    x_values.append(key)
                    y_values.append(1)
        return x_values, y_values


if __name__ == '__main__':
    variable_definitions = [
        KeyboardTypingDurationExtractor(),
        KeyboardKeyCountExtractor(),
        KeyboardPressedDurationExtractor()
    ]

    variable_mapping = {
        variable.name: variable for variable in variable_definitions
    }

    enrollment = os.path.join("..", "..", "data", "participant_enrollment.csv")
    condition_definitions = HandednessExtractor(enrollment)

    processed = {}

    for psydat_file in tqdm(psydat_files):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        for extractor in variable_definitions:
            predictor_name = extractor.name
            predictor_keys, predictor_values = extractor.process(parser)
            assert len(predictor_keys) == len(predictor_values), "Size doesn't match!"
            predictor_associated_values = []
            predictor_associated_vars = []
            predictor_keys = np.array(predictor_keys)
            predictor_values = np.array(predictor_values)

            if "name" not in processed:
                processed["name"] = []
            processed["name"].extend([predictor_name] * len(predictor_values))
            if "handedness" not in processed:
                processed["handedness"] = []
            processed["handedness"].extend([condition_definitions.process(participant_id)] * len(predictor_values))
            if "key" not in processed:
                processed["key"] = []
            processed["key"].extend(predictor_keys)
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

