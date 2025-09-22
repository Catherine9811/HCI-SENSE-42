import numpy as np
import os
from tqdm import tqdm
import mne


def compute_eeg_power_band(raw, fmin=0.0, fmax=100.0, tres=5.0):
    # There is two ways of solving the power density of a specific band
    # You either take the mean over the spectrum or you take the sum of the spectrum normalized by the absolute power
    # The reason that you have to normalize it when taking the sum
    # is because we need to account for the band width difference
    duration = raw.times[-1]
    psd_list, freq_list = [], []
    for twindow in np.arange(0, (duration // tres) * tres, tres):
        psds, freqs = raw.compute_psd(
            method='welch',
            fmin=fmin, fmax=fmax,
            tmin=twindow, tmax=twindow+tres,
        ).get_data(return_freqs=True)
        psd_list.append(psds)
        freq_list.append(freqs)
    return {
        'psds': np.stack(psd_list),
        'freqs': np.stack(freq_list)
    }


participants = list(range(1, 43))

save_psd = True
saving_psd_dir = os.path.join('data', 'psd_welch_1min')
time_resolution = 60.0

for participant_id in (bar := tqdm(participants)):
    bar.set_description(f"{participant_id}")
    # Load the .set file
    eeglab_file = rf'D:\HCI PROCESSED DATA\CleanedEEGLab\P{participant_id:03d}.set'
    raw = mne.io.read_raw_eeglab(eeglab_file, preload=True, verbose="WARNING")
    raw = raw.set_eeg_reference("average")
    psd_dict = compute_eeg_power_band(raw, fmin=0.0, fmax=100.0, tres=time_resolution)
    if save_psd:
        if not os.path.exists(saving_psd_dir):
            os.makedirs(saving_psd_dir)
        np.savez_compressed(os.path.join(saving_psd_dir, f"P{participant_id:03d}-psd.npz"),
                            channels=raw.ch_names,
                            resolution=time_resolution,
                            **psd_dict)
