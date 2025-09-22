import matplotlib.pyplot as plt
import matplotlib
import addcopyfighandler
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import re
import mne
from scipy import stats

import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

import seaborn as sns
from mne.viz import iter_topography
from numpy.polynomial.polynomial import polyfit
from sklearn.linear_model import LinearRegression

from mne_utils import snr_spectrum, plot_psd_snr
from common_variables import *

from event_loader import Loader


def subplot2grid(shape, loc, rowspan=1, colspan=1, width_ratios=None):
    fig = plt.gcf()
    nrows, ncols = shape
    row, col = loc
    gs = fig.add_gridspec(nrows, ncols, width_ratios=width_ratios)
    return fig.add_subplot(gs[row:row + rowspan, col:col + colspan])


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


def remappedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax)
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin))
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin))
    '''

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False),
        np.linspace(0.5, stop, 129)
    ])

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


event_path = os.path.join('data', 'events')

variables = (
    # {
    #     "name": "All",
    #     "field": ("[minute * 60 for minute in range(len(everything))]", "everything"),
    #     "z": 5
    # },
    # {
    #     "name": "Gamma",
    #     "field": ("[minute * 60 for minute in range(len(gamma))]", "gamma"),
    #     "z": 5
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

local_filter = False
visual_filter = False
save_csv = False
norm_by_kss = False
psd_dir = os.path.join('data', 'psd_welch_1min')
info_file = os.path.join('data', 'default-info.fif.gz')
info = mne.io.read_info(info_file)

channels = []
variable_name_y = predictor["name"]
variable_field_y = predictor["field"]
variable_type = predictor["type"]
variable_max = predictor["max"]
cluster_test_result = {variable["name"]: None for variable in variables}


def linear_regression_obsolete(*args, participant_ids=None):
    n_classes = len(args)
    n_samples_per_class = np.array([len(a) for a in args])
    n_samples = np.sum(n_samples_per_class)
    class_values = []
    for value, count in enumerate(n_samples_per_class):
        class_values.extend([value + 1] * count)
    class_values = np.array(class_values)
    outcome_values = np.zeros((len(channels)))
    for i, channel_name in enumerate(channels):
        channel_values = []
        for arg in args:
            channel_values.extend(arg[:, i])

        channel_values = np.array(channel_values)
        channel_values = 10 * np.log10(channel_values) + 120
        x = sm.add_constant(class_values)
        model = sm.OLS(channel_values, x)
        result = model.fit()
        outcome_values[i] = result.fvalue # result.tvalues[1] # result.params[1] * 1e14 # result.fvalue
    return outcome_values


def linear_regression_per_subject(*args, participant_ids=None):
    """
    Runs an OLS regression for each participant separately and then averages F-values across participants.
    Args:
        *args: list of arrays [n_samples, n_channels] for each group
        participant_ids: list of subject IDs aligned with samples in args
    Returns:
        outcome_values: array of average F-values across participants for each channel
    """
    unique_subjects = np.unique(np.concatenate(participant_ids))
    n_channels = len(channels)
    outcome_values = np.zeros(n_channels)

    # Loop over channels
    for ch in range(n_channels):
        fvals = []

        # Loop over participants
        for subj in unique_subjects:
            subj_data = []
            subj_groups = []

            # Collect data for this participant across groups
            for group_idx, (group_data, subj_ids) in enumerate(zip(args, participant_ids)):
                mask = np.array(subj_ids) == subj
                if np.any(mask):
                    subj_data.extend(group_data[mask, ch])
                    subj_groups.extend([group_idx + 1] * np.sum(mask))

            if len(subj_data) > 1 and len(np.unique(subj_groups)) > 1:
                y = 10 * np.log10(np.array(subj_data)) + 120
                X = sm.add_constant(np.array(subj_groups))
                model = sm.OLS(y, X)
                try:
                    result = model.fit()
                    fvals.append(result.fvalue)
                except Exception:
                    continue

        # Average F across participants for this channel
        outcome_values[ch] = np.mean(fvals) if len(fvals) > 0 else 0.0

    return outcome_values


def linear_regression(*args, participant_ids=None):
    n_classes = len(args)
    n_samples_per_class = np.array([len(a) for a in args])
    class_values = []
    subject_ids = []

    # Expand group labels and subject ids
    for class_idx, (group_data, subj_ids) in enumerate(zip(args, participant_ids)):
        class_values.extend([class_idx + 1] * len(group_data))
        subject_ids.extend(subj_ids)

    class_values = np.array(class_values)
    subject_ids = np.array(subject_ids)

    outcome_values = np.zeros((len(channels)))

    for ch, ch_name in enumerate(channels):
        channel_data = []
        for group in args:
            channel_data.extend(group[:, ch])
        channel_data = np.array(channel_data)

        df = pd.DataFrame({
            'signal': 10 * np.log10(channel_data) + 120,
            'group': class_values,
            'subject': subject_ids
        })

        try:
            model = sm.MixedLM.from_formula("signal ~ group", groups="subject", data=df)
            result = model.fit()
            f_stat = result.tvalues['group'] ** 2  # approximate F from t^2
            outcome_values[ch] = f_stat
        except Exception as e:
            outcome_values[ch] = 0  # fallback on failure
            print(f"Warning: channel {ch} failed to fit mixed model. Error: {e}")

    return outcome_values


def linear_regression_get_beta(*args, participant_ids=None):
    n_classes = len(args)
    n_samples_per_class = np.array([len(a) for a in args])
    class_values = []
    subject_ids = []

    # Expand group labels and subject ids
    for class_idx, (group_data, subj_ids) in enumerate(zip(args, participant_ids)):
        class_values.extend([class_idx + 1] * len(group_data))
        subject_ids.extend(subj_ids)

    class_values = np.array(class_values)
    subject_ids = np.array(subject_ids)

    outcome_values = np.zeros((len(channels)))

    for ch, ch_name in enumerate(channels):
        channel_data = []
        for group in args:
            channel_data.extend(group[:, ch])
        channel_data = np.array(channel_data)

        df = pd.DataFrame({
            'signal': 10 * np.log10(channel_data) + 120,
            'group': class_values,
            'subject': subject_ids
        })

        try:
            model = sm.MixedLM.from_formula("signal ~ group", groups="subject", data=df)
            result = model.fit()
            # Get slope (beta coefficient) for 'group'
            beta = result.params['group']  # <-- this is the slope estimate
            outcome_values[ch] = beta
        except Exception as e:
            outcome_values[ch] = 0  # fallback on failure
            print(f"Warning: channel {ch} failed to fit mixed model. Error: {e}")

    return outcome_values


def linear_regression_get_beta_obsolete(*args, participant_ids=None):
    n_classes = len(args)
    n_samples_per_class = np.array([len(a) for a in args])
    n_samples = np.sum(n_samples_per_class)
    class_values = []
    for value, count in enumerate(n_samples_per_class):
        class_values.extend([value + 1] * count)
    class_values = np.array(class_values)
    outcome_values = np.zeros((len(channels)))
    for i, channel_name in enumerate(channels):
        channel_values = []
        for arg in args:
            channel_values.extend(arg[:, i])

        channel_values = np.array(channel_values)
        channel_values = 10 * np.log10(channel_values) + 120
        x = sm.add_constant(class_values)
        model = sm.OLS(channel_values, x)
        result = model.fit()
        outcome_values[i] = result.params[1] # result.fvalue
    return outcome_values


pval = 0.05  # arbitrary
dfn = 9 - 1  # degrees of freedom numerator
dfd = 41 - 9  # degrees of freedom denominator
thresh = stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)
print(thresh)

for variable in tqdm(variables):
    variable_name_x = variable["name"]
    variable_field_x = variable["field"]
    full_variable_x = []
    full_variable_y = []
    full_variable_p = []
    for participant_index, participant_id in enumerate(range(2, 43)):
        if participant_id not in [2, 6, 8, 19, 25, 26, 29, 30, 31, 33, 34, 39, 40, 42]:
            continue
        # if participant_id not in [2, 6, 8, 19, 25]:
        #     continue
        # mne.viz.plot_sensors(info, ch_type='eeg', axes=plt.gca(), show_names=False, pointsize=0)
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

        if local_filter:
            invalid_mask = corr_x > 8 * np.median(corr_x)
            invalid_mask = np.any(invalid_mask, axis=1)
            print(invalid_mask.sum())
            corr_x = corr_x[~invalid_mask]
            corr_y = corr_y[~invalid_mask]

        full_variable_x.extend(corr_x)
        full_variable_y.extend(corr_y)
        full_variable_p.extend([participant_id for _ in corr_x])

    flat_variable_x = np.array(full_variable_x)
    flat_variable_y = np.array(full_variable_y)
    flat_variable_p = np.array(full_variable_p)

    X = []
    P = []
    for i in range(variable_max):
        mask = flat_variable_y == (i + 1)
        data = flat_variable_x[:, :][mask]
        X.append(data)
        P.append(flat_variable_p[mask])

    F_obs, clusters, cluster_pv, H0 = \
        mne.stats.permutation_cluster_test(X,
                                           stat_fun=lambda *args: linear_regression_per_subject (*args, participant_ids=P),
                                           threshold=thresh, adjacency=mne.channels.find_ch_adjacency(info, 'eeg')[0], out_type='mask')
    print(F_obs, clusters, cluster_pv)
    F_obs = linear_regression_get_beta(*X, participant_ids=P)
    cluster_test_result[variable_name_x] = (F_obs, clusters, cluster_pv)

total_clusters = 1
vmin = np.inf
vmax = 0
cmap = "coolwarm"
for eeg_band in cluster_test_result:
    total_clusters += len(cluster_test_result[eeg_band][1])
    vmin = min(vmin, np.min(cluster_test_result[eeg_band][0]))
    vmax = max(vmax, np.max(cluster_test_result[eeg_band][0]))
maxval = max(abs(vmin), abs(vmax))
vmin = -maxval
vmax = maxval
# cmap = remappedColorMap(cmap, 0, abs(vmin / (vmax - vmin)), 1.0)
fig, ax = plt.subplots(1, total_clusters, width_ratios=[30] * (total_clusters - 1) + [1])
if total_clusters == 1:
    ax = [ax]

current_ax = 0
for eeg_band in cluster_test_result:
    F_obs, clusters, cluster_pv = cluster_test_result[eeg_band]
    for index, (cluster, p_value) in enumerate(zip(clusters, cluster_pv)):
        data = np.array(F_obs)
        evoked = mne.EvokedArray(data[..., np.newaxis], info)
        if current_ax == total_clusters - 2:
            colorbar = True
            axes = [ax[current_ax], ax[-1]]
        else:
            colorbar = False
            axes = ax[current_ax]
        mne.viz.plot_evoked_topomap(evoked, times=0, ch_type='eeg', colorbar=colorbar, show_names=True,
                                    units="Slope (Beta)\n[dB/unit]",
                                    time_format="",
                                    # image_interp='nearest',
                                    mask=cluster[..., np.newaxis],
                                    vlim=(vmin, vmax),
                                    mask_params=dict(marker='o',
                                                     markerfacecolor='w',
                                                     markeredgecolor='k',
                                                     alpha=0.5,
                                                     linewidth=0,
                                                     markersize=35),
                                    scalings=1.0, cbar_fmt='%1.2f', cmap=cmap, axes=axes)
        ax[current_ax].set_title(f"{eeg_band} Band")
        ax[current_ax].set_ylabel(f"Cluster {index + 1}")
        ax[current_ax].set_xlabel(f"p_value={p_value:1.4f}")
        current_ax += 1
ax[-1].set_ylim(vmin, vmax)
fig.suptitle(f"Significant EEG Clusters via Permutation Test as Predictors for {variable_name_y}")
plt.tight_layout()
plt.show()
