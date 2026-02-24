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


class PerformanceAnswerExtractor:
    name = "performance"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "performance:_how_successful_were_you_in_accomplishing_what_you_were_asked_to_do_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        t_values = [entry["browser_content.started"] for entry in questionnaires]
        return x_values, y_values, t_values


class TemporalDemandAnswerExtractor:
    name = "temporal_demand"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "temporal_demand:_how_hurried_or_rushed_was_the_pace_of_the_task_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        t_values = [entry["browser_content.started"] for entry in questionnaires]
        return x_values, y_values, t_values


class AttentivenessAnswerExtractor:
    name = "attentiveness"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "attentiveness:_how_focused_were_you_on_performing_the_task_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        t_values = [entry["browser_content.started"] for entry in questionnaires]
        return x_values, y_values, t_values


class SleepinessAnswerExtractor:
    name = "sleepiness"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "sleepiness:_how_sleepy_are_you_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        t_values = [entry["browser_content.started"] for entry in questionnaires]
        return x_values, y_values, t_values


class EffortAnswerExtractor:
    name = "effort"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "effort:_how_hard_did_you_have_to_work_to_accomplish_your_level_of_performance_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        t_values = [entry["browser_content.started"] for entry in questionnaires]
        return x_values, y_values, t_values


class FrustrationAnswerExtractor:
    name = "frustration"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "frustration:_how_insecure__discouraged__irritated__stressed__and_annoyed_were_you_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        t_values = [entry["browser_content.started"] for entry in questionnaires]
        return x_values, y_values, t_values


class MentalDemandAnswerExtractor:
    name = "mental_demand"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "mental_demand:_how_mentally_demanding_was_the_task_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        t_values = [entry["browser_content.started"] for entry in questionnaires]
        return x_values, y_values, t_values


if __name__ == '__main__':
    predictor_definitions = [
        PerformanceAnswerExtractor(),
        TemporalDemandAnswerExtractor(),
        AttentivenessAnswerExtractor(),
        SleepinessAnswerExtractor(),
        EffortAnswerExtractor(),
        FrustrationAnswerExtractor(),
        MentalDemandAnswerExtractor()
    ]
    processed = {}

    for psydat_file in tqdm(psydat_files):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        for extractor in predictor_definitions:
            predictor_name = extractor.name
            predictor_times, predictor_values, predictor_initiation = extractor.process(parser)
            predictor_times = np.array(predictor_times)
            predictor_values = np.array(predictor_values)
            predictor_initiation = np.array(predictor_initiation)
            if "name" not in processed:
                processed["name"] = []
            processed["name"].extend([predictor_name] * len(predictor_values))
            if "time" not in processed:
                processed["time"] = []
            processed["time"].extend(predictor_times)
            if "value" not in processed:
                processed["value"] = []
            processed["value"].extend(predictor_values)
            if "initiation" not in processed:
                processed["initiation"] = []
            processed["initiation"].extend(predictor_initiation)
            if "participant" not in processed:
                processed["participant"] = []
            processed["participant"].extend([participant_id] * len(predictor_values))

    # processed = filter_nan_indices(processed)

    # Convert processed dictionary to DataFrame
    df = pd.DataFrame(processed)

    save_directory = os.path.join("processed_data", "questionnaire")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save to CSV file
    df.to_csv(os.path.join(save_directory, f"{len(psydat_files)}-questionnaires.csv"), index=False)

    print("Processed data saved.")

