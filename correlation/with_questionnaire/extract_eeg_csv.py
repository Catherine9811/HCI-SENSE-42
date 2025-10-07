import numpy as np
import pickle
import os
import pandas as pd
import jellyfish
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser
from data_definition import psydat_files
from analyze_eeg.common_variables import eeg_bands, frequency_bands

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


class EEGBandPowerExtractor:
    name = eeg_bands

    def process(self, parser):
        task = parser["mail_homescreen"]
        # Extracting values for plotting
        timestamps = [entry["mail_homescreen.started"] for entry in task]
        participant_id = int(parser.id())
        x_values = []
        y_values = []
        for index, time in enumerate(timestamps):
            event_path = os.path.join("..", "..", "analyze_eeg", "data", "psd_welch_event7to9",
                                      f"P{participant_id:03d}-psd-event7to9-{index+1:02d}.npz")
            if not os.path.exists(event_path):
                continue
            data = np.load(event_path)
            psds = data["psds"]
            freqs = data["freqs"].squeeze()
            mean_psd = psds.mean(axis=0)
            y_value = []
            for band in eeg_bands:
                fmin, fmax = frequency_bands[band]
                idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
                if len(idx) > 0:
                    power = mean_psd[idx].mean()
                else:
                    power = np.nan
                y_value.append(10 * np.log10(power) + 120)
            x_values.append(time)
            y_values.append(y_value)
        return x_values, y_values


class MentalDemandAnswerExtractor:
    name = "mental_demand"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "mental_demand:_how_mentally_demanding_was_the_task_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class EffortAnswerExtractor:
    name = "effort"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "effort:_how_hard_did_you_have_to_work_to_accomplish_your_level_of_performance_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class FrustrationAnswerExtractor:
    name = "frustration"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "frustration:_how_insecure__discouraged__irritated__stressed__and_annoyed_were_you_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


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


if __name__ == '__main__':
    predictor_definition = EEGBandPowerExtractor()

    outcome_definition = SleepinessAnswerExtractor()

    processed = {}

    for psydat_file in tqdm(psydat_files):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        outcome_times, outcome_values = outcome_definition.process(parser)
        outcome_times, outcome_values = filter_time_series(outcome_times, outcome_values)
        for extractor in [predictor_definition]:
            predictor_name = extractor.name
            predictor_times, predictor_values = extractor.process(parser)
            predictor_associated_values = []
            predictor_associated_times = []
            predictor_times = np.array(predictor_times)
            predictor_values = np.array(predictor_values)
            for outcome_time in outcome_times:
                associated_values = predictor_values[
                    (predictor_times > outcome_time - SMOOTH_WINDOW) & (predictor_times <= outcome_time)]
                predictor_associated_values.append(np.mean(associated_values, axis=0))
                predictor_associated_times.append(np.mean(predictor_times))
            for predictor_index, predictor_name in enumerate(extractor.name):
                if predictor_name not in processed:
                    processed[predictor_name] = []
                processed[predictor_name].extend(np.array(predictor_associated_values)[:, predictor_index])
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

    save_directory = os.path.join("processed_data", "event7to9")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save to CSV file
    df.to_csv(os.path.join(save_directory, f"{len(psydat_files)}-{outcome_definition.name}.csv"), index=False)

    print("Processed data saved.")

