import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from data_parser import DataParser
from data_definition import psydat_files


time_spent = {}
for i, file_path in tqdm(enumerate(psydat_files)):
    parser = DataParser(f"../data/{file_path}")
    for task_name, task_key in [
        ('Dragging Files', ['file_manager_dragging']),
        ('Clicking Files', ['file_manager_opening']),
        ('Closing Windows', 'window_close'),
        ('Shadow Typing', 'mail_content'),
        ('Side-by-side Typing', 'notes_repeat'),
        ('Browser', ['browser_navigation', 'browser_content']),
        ('Empty Trash Bin', ['trash_bin_select', 'trash_bin_confirm']),
        ('Opening Application', ['mail_notification', 'file_manager_homescreen', 'trash_bin_homescreen', 'notes_homescreen', 'browser_homescreen'])
    ]:
        if task_name not in time_spent:
            time_spent[task_name] = []
        if isinstance(task_key, str):
            trials = parser[task_key]
            for trial in trials:
                time_spent[task_name].append(trial[f"{task_key}.stopped"] - trial[f"{task_key}.started"])
        else:
            trial_group = {}
            for sub_key in task_key:
                trials = parser[sub_key]
                for trial in trials:
                    if trial["trials.thisN"] not in trial_group:
                        trial_group[trial["trials.thisN"]] = 0
                    trial_group[trial["trials.thisN"]] += trial[f"{sub_key}.stopped"] - trial[f"{sub_key}.started"]
            time_spent[task_name].extend(list(trial_group.values()))

for task, count in time_spent.items():
    print(task, np.mean(count), np.std(count))
