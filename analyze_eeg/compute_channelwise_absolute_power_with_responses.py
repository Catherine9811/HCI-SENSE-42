import matplotlib.pyplot as plt
import matplotlib
import addcopyfighandler
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import re
import mne

from common_variables import *

from event_loader import Loader

def mask_outliers(data, z_threshold=5):
    # Calculate the mean and standard deviation of the data
    mean_value = np.median(data, axis=1, keepdims=True)
    std_dev = np.std(data, axis=1, keepdims=True)

    # Calculate Z-scores for each data point
    z_scores = (data - mean_value) / std_dev

    # Identify outliers based on Z-scores
    mask = np.abs(z_scores) > z_threshold

    return mask


def compute_eeg_power_band(psds, freqs, fmin=1.0, fmax=100.0):
    # There is two ways of solving the power density of a specific band
    # You either take the mean over the spectrum or you take the sum of the spectrum normalized by the absolute power
    # The reason that you have to normalize it when taking the sum
    # is because we need to account for the band width difference
    frequency_data = {name: [] for name in frequency_bands}
    # Normalize the PSDs
    frequency_index = np.logical_and(freqs[0] >= fmin, freqs[0] <= fmax)
    # psds /= np.sum(psds[:, :, frequency_index], axis=2, keepdims=True)
    for name in frequency_bands:
        fmin_i, fmax_i = frequency_bands[name]
        idx_band = np.logical_and(freqs[0] >= fmin_i, freqs[0] <= fmax_i)
        psds_band = psds[:, :, idx_band].mean(axis=-1)
        frequency_data[name] = psds_band
    # frequency_data = {name: np.stack(frequency_data[name]) for name in frequency_data}

    return frequency_data


event_path = os.path.join('data', 'events')

variables = (
    # {
    #     "name": "All",
    #     "field": ("[minute * 60 for minute in range(len(everything))]", "everything"),
    # },
    # {
    #     "name": "Gamma",
    #     "field": ("[minute * 60 for minute in range(len(gamma))]", "gamma"),
    # },

    {
        "name": "Delta",
        "field": ("[minute * 60 for minute in range(len(delta))]", "delta"),
    },
    {
        "name": "Theta",
        "field": ("[minute * 60 for minute in range(len(theta))]", "theta"),
    },
    {
        "name": "Alpha",
        "field": ("[minute * 60 for minute in range(len(alpha))]", "alpha"),
    },
    {
        "name": "Beta",
        "field": ("[minute * 60 for minute in range(len(beta))]", "beta"),
    },
)

predictor = {
    "name": "Sleepiness Score",
    "type": "sleepiness",
    "max": 9,
    "field": ("response_times", "response_value"),
}

# predictor = {
#     "name": "Attentiveness Score",
#     "type": "attentiveness",
#     "max": 7,
#     "field": ("response_times", "response_value"),
# }

# predictor = {
#     "name": "Performance Score",
#     "type": "performance",
#     "max": 7,
#     "field": ("response_times", "response_value"),
# }


def time_filter(seconds):   # 0 <= seconds <= ~3600
    return True


smooth_window = 5 * 60  # seconds

save_csv = True
csv_path = os.path.join('data', f'channelwise_absolute_power_with_{predictor["type"]}.csv')
psd_dir = os.path.join('data', 'psd_welch_1min')
info_file = os.path.join('data', 'default-info.fif.gz')
info = mne.io.read_info(info_file)

channels = []
variable_name_y = predictor["name"]
variable_field_y = predictor["field"]
variable_type = predictor["type"]
variable_max = predictor["max"]
all_variables = {
    variable["name"]: {
        "value": [],
        "response": [],
        "participant_id": [],
        "time": []
    }
    for variable in variables
}

for participant_index, participant_id in tqdm(enumerate(range(2, 43))):
    conditioned_files = [os.path.join(event_path, name) for name in os.listdir(event_path)
                         if name == f"P{participant_id:03d}.txt"]
    loader = Loader(conditioned_files, type=variable_type,
                    filtering=time_filter)
    response_times, response_value = loader.read()
    assert len(response_times) > 0, f"P{participant_id:03d} missing {variable_type}!"
    # Load the .set file
    psd_file = rf'{psd_dir}\P{participant_id:03d}-psd.npz'
    psd_dict = np.load(psd_file)
    channels = psd_dict['channels']
    resolution = psd_dict['resolution']
    psds = psd_dict['psds']
    freqs = psd_dict['freqs']
    assert all(a == b for a, b in zip(channels, info.ch_names)), "Channel names/orders mismatch!"

    locals().update(compute_eeg_power_band(psds, freqs))
    eeg_freqs = {}
    for eeg_band_name in eeg_bands:
        eeg_freqs[eeg_band_name] = eval(eeg_band_name)  # Shape (T, C)

    locals().update({name: eeg_freqs[name] for name in eeg_freqs})

    for variable in variables:
        variable_name_x = variable["name"]
        variable_field_x = variable["field"]
        full_variable_x = all_variables[variable_name_x]["value"]
        full_variable_y = all_variables[variable_name_x]["response"]
        full_variable_p = all_variables[variable_name_x]["participant_id"]
        full_variable_t = all_variables[variable_name_x]["time"]
        variable_x = np.array(eval(variable_field_x[1]))
        variable_x_t = np.array(eval(variable_field_x[0])).astype(float)
        variable_y = np.array(eval(variable_field_y[1])).astype(float)
        variable_y_t = np.array(eval(variable_field_y[0]))

        # Align time stamps
        corr_y = variable_y
        corr_t = variable_y_t
        corr_x = []
        for timestamp in corr_t:
            smoothed_x = variable_x[(variable_x_t > timestamp - smooth_window) & (variable_x_t <= timestamp)]
            corr_x.append(np.mean(smoothed_x, axis=0))
        corr_x = np.array(corr_x)

        full_variable_x.extend(corr_x)
        full_variable_y.extend(corr_y)
        full_variable_t.extend(corr_t)
        full_variable_p.extend([participant_id for _ in corr_x])

all_variables_length = [
    [
        len(all_variables[variable["name"]][key]) for key in ["value", "response", "participant_id"]
    ]
    for variable in variables
]
assert len(np.unique(np.array(all_variables_length).flatten())) == 1, "Length not matching between variables"

if save_csv:
    rows = []
    first_variable = all_variables[variables[0]["name"]]

    for index in range(len(first_variable["participant_id"])):
        participant_id = first_variable["participant_id"][index]
        response = first_variable["response"][index]
        time = first_variable["time"][index]
        for variable in variables:
            band = variable["name"].lower()
            all_powers = all_variables[variable["name"]]["value"]

            for ch_index, channel in enumerate(channels):
                power = all_powers[index][ch_index]

                rows.append({
                    "participant_id": participant_id,
                    "response": response,
                    "band": band,
                    "power": power,
                    "channel": channel,
                    "time": time
                })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)