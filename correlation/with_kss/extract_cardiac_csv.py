import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import addcopyfighandler
import mne
from collections import defaultdict
from data_parser import DataParser
from correlation.with_kss.extract_behavioural_csv import \
    SleepinessAnswerExtractor, AttentivenessAnswerExtractor, PerformanceAnswerExtractor, TemporalDemandAnswerExtractor
from data_definition import psydat_files


SMOOTH_WINDOW = 5 * 60


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


class CardiacRRIntervalExtractor:
    name = "cardiac_rr_interval"

    def process(self, ecg):
        # Detect ECG events (R-waves) in lead II using find_ecg_events
        # Note: find_ecg_events returns three values; here we only need the first one which contains event positions.
        ecg_events, ch_ecg, average_pulse = mne.preprocessing.find_ecg_events(ecg, ch_name='ECG2')

        # Plot the derived 3-lead ECG signals
        # ecg.plot(title='Derived 3-lead ECG signals', events=ecg_events, duration=10, n_channels=3)

        # Compute event times in seconds
        sfreq = ecg.info['sfreq']
        event_times = ecg_events[:, 0] / sfreq

        # Compute R-R intervals (the time difference between consecutive R-waves)
        rr_intervals = np.diff(event_times)
        # Use the midpoint of each interval for the time axis
        rr_time = (event_times[:-1] + event_times[1:]) / 2

        # Clean the data
        min_bpm = 50
        max_bpm = 100
        valid_intervals = np.logical_and(rr_intervals >= 60.0 / max_bpm, rr_intervals <= 60.0 / min_bpm)
        y_values = rr_intervals[valid_intervals]
        x_values = rr_time[valid_intervals]

        return x_values, y_values


if __name__ == '__main__':
    predictor_definitions = [
        CardiacRRIntervalExtractor()
    ]

    outcome_definition = SleepinessAnswerExtractor()

    processed = {}

    for psydat_file in tqdm(psydat_files):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        outcome_times, outcome_values = outcome_definition.process(parser)
        ecg_filename = os.path.join("..", "..", "data", "ECG", f"P{participant_id:03}.fif")
        if not os.path.exists(ecg_filename):
            continue
        ecg_file = mne.io.read_raw_fif(ecg_filename, preload=True)

        for extractor in predictor_definitions:
            predictor_name = extractor.name
            predictor_times, predictor_values = extractor.process(ecg_file)
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

    save_directory = os.path.join("processed_data", "cardiac")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save to CSV file
    df.to_csv(os.path.join(save_directory, f"{len(psydat_files)}-{outcome_definition.name}.csv"), index=False)

    print("Processed data saved.")

