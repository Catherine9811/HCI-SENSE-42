import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from data_parser import DataParser
from data_definition import psydat_files


num_trials = []
len_trials = []
num_styles = {}
for file_path in tqdm(psydat_files):
    parser = DataParser(f"../data/{file_path}")
    start_key = "style_randomizer"
    end_key = "loop_end"
    trial_starts = parser[start_key]
    trial_ends = parser[end_key]
    stylizer = parser["operating_system_style"]
    for entry in stylizer:
        if 'trials.thisN' in entry:
            style = entry["operating_system_style"]
            if style not in num_styles:
                num_styles[style] = 0
            num_styles[style] += 1
    if len(trial_starts) != len(trial_ends):
        continue

    num_trials.append(len(trial_ends))

    for start, end in zip(trial_starts, trial_ends):
        assert start["trials.thisN"] == end["trials.thisN"]
        len_trials.append(end[f"{end_key}.started"] - start[f"{start_key}.started"])

print("Number of Trials", np.mean(num_trials), np.std(num_trials))
print("Length of Trials", np.mean(len_trials), np.std(len_trials))
for style, count in num_styles.items():
    print(style, count)
