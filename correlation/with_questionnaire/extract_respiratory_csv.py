import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from analyze_respiratory.rip import Resp
import datetime
from data_parser import DataParser
from correlation.with_questionnaire.extract_behavioural_csv import \
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


class RespiratoryInhalationDurationExtractor:
    name = "respiratory_inhalation_duration"

    def process(self, resp):
        x_values = [i.start_time for i in resp.inhalations]
        y_values = [i.duration() for i in resp.inhalations]
        return x_values, y_values


class RespiratoryExhalationDurationExtractor:
    name = "respiratory_exhalation_duration"

    def process(self, resp):
        x_values = [i.start_time for i in resp.exhalations]
        y_values = [i.duration() for i in resp.exhalations]
        return x_values, y_values


if __name__ == '__main__':
    predictor_definitions = [
        RespiratoryInhalationDurationExtractor(),
        RespiratoryExhalationDurationExtractor()
    ]

    outcome_definition = SleepinessAnswerExtractor()

    processed = {}

    for psydat_file in tqdm(psydat_files):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        outcome_times, outcome_values = outcome_definition.process(parser)
        resp = Resp.from_wav(os.path.join("..", "..", "data", "Respiration", f"P{participant_id:03}.wav"))

        baseline = resp.baseline_savgol(60)
        resp.remove_baseline(method='savgol')
        resp.find_cycles(include_holds=True)

        for extractor in predictor_definitions:
            predictor_name = extractor.name
            predictor_times, predictor_values = extractor.process(resp)
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

    save_directory = os.path.join("processed_data", "respiratory")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save to CSV file
    df.to_csv(os.path.join(save_directory, f"{len(psydat_files)}-{outcome_definition.name}.csv"), index=False)

    print("Processed data saved.")

