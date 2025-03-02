import mne
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler
from scipy.io import wavfile

# Define the file path
participant_id = "P002"
bdf_file = rf"../data/EEG/{participant_id}.bdf"

channel = ['Resp']

# Load the raw BDF file
raw = mne.io.read_raw_bdf(bdf_file, preload=True)

print(raw.info['ch_names'])

raw = raw.pick(picks=channel)

# Resample frequency
raw = raw.resample(32)

mne.viz.plot_raw_psd(raw, picks=channel, fmax=10)
mne.viz.plot_raw(raw)

plt.show()

data = raw.get_data(picks=channel)
print(raw.info['sfreq'])
wavfile.write(rf"../data/Respiration/{participant_id}.wav", int(raw.info['sfreq']), data[0])
