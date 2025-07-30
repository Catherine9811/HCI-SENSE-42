import os
import numpy as np


def equal(event, code):
    return event == code
    # return (event & code) == code # disabled since we are not following strict binary assignments


# Questions
QUESTION_BASE = 100
QUESTION_LEAP = 10
question_index = {
    "sleepiness": 0,
    "mental_demand": 1,
    "temporal_demand": 2,
    "performance": 3,
    "effort": 4,
    "frustration": 5,
    "attentiveness": 6
}
question_base = {key: QUESTION_BASE + QUESTION_LEAP * index for key, index in question_index.items()}
# texts = [
#     "Sleepiness: How sleepy are you?",
#     "Mental Demand: How mentally demanding was the task?",
#     #"Physical Demand: How physically demanding was the task?",
#     "Temporal Demand: How hurried or rushed was the pace of the task?",
#     "Performance: How successful were you in accomplishing what you were asked to do?",
#     "Effort: How hard did you have to work to accomplish your level of performance?",
#     "Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?",
#     "Attentiveness: How focused were you on performing the task?"
# ]
# ## Answered questions will be encoded as
# QUESTION_BASE + QUESTION_LEAP * QUESTION_INDEX + QUESTION_RATING


class Loader:
    def __init__(self, file_path="", type="sleepiness",
                 filtering=None):
        # Read the original file
        self.file_path = file_path

        assert type in question_index, "Unsupported type!"

        # Define the Hz
        self.event_min = question_base[type]
        self.event_max = self.event_min + QUESTION_LEAP

        self.time_filter = filtering

    def read(self):
        response_times = []
        response_value = []

        files_list = self.file_path
        if not isinstance(self.file_path, list):
            files_list = [self.file_path]

        for filepath in files_list:
            with open(filepath, 'r') as original_file:
                lines = original_file.readlines()

            # Iterate through the original lines and check event pairs
            for index, line in enumerate(lines):
                # Split the line into columns
                columns = line.strip().split('\t')
                if index == 0:
                    continue
                current_timestamp = float(columns[2])   # Time in s
                if self.time_filter is not None and not self.time_filter(current_timestamp):
                    continue
                event_code = int(float(columns[1]))

                # Check if it's an sleepiness questionaire line
                if self.event_min <= event_code < self.event_max:
                    onset = current_timestamp  # Time in s
                    score = int(event_code - self.event_min)
                    response_times.append(onset)
                    response_value.append(score)
        # sleepiness_reaction_y = [item % self.sleepiness_interval for item in sleepiness_reaction_y]
        # sleepiness_y = [y / 9 for y in sleepiness_y]
        return response_times, response_value
