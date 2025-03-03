import mne
import numpy as np
import matplotlib.pyplot as plt

# Define the file path
participant_id = "P002"
bdf_file = rf"../data/EEG/{participant_id}.bdf"
fif_file = rf"../data/ECG/{participant_id}.fif"

# Read the BDF file (replace 'your_file.bdf' with the actual file path)
raw = mne.io.read_raw_bdf(bdf_file, preload=True)

# Define LA, RA, LL channels
channel_la = "EXG3"
channel_ra = "EXG4"
channel_ll = "EarL"

# Extract the raw data for LA, RA, and LL channels
data_la = raw.get_data(picks=[channel_la])
data_ra = raw.get_data(picks=[channel_ra])
data_ll = raw.get_data(picks=[channel_ll])

# Compute the derived 3-lead ECG signals:
# Lead I = LA - RA
# Lead II = LL - RA
# Lead III = LL - LA
ecg1 = data_la - data_ra  # Lead I
ecg2 = data_ll - data_ra  # Lead II
ecg3 = data_ll - data_la  # Lead III

# Create a new Raw object for the derived ECG signals
sfreq = raw.info['sfreq']  # Sampling frequency from the original raw data
ch_names = ['ECG1', 'ECG2', 'ECG3']
ch_types = ['ecg', 'ecg', 'ecg']

# Create the info structure for the new raw object
info_ecg = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Stack the computed ECG data (shape: n_channels x n_samples)
data_ecg = np.vstack([ecg1, ecg2, ecg3])
raw_ecg = mne.io.RawArray(data_ecg, info_ecg)

# Save the new raw ECG object to a FIF file
raw_ecg.save(fif_file, overwrite=True)

# Detect ECG events (R-waves) in lead II using find_ecg_events
# Note: find_ecg_events returns three values; here we only need the first one which contains event positions.
ecg_events, ch_ecg, average_pulse = mne.preprocessing.find_ecg_events(raw_ecg, ch_name='ECG2')

# Plot the derived 3-lead ECG signals
raw_ecg.plot(title='Derived 3-lead ECG signals', events=ecg_events, duration=10, n_channels=3)
