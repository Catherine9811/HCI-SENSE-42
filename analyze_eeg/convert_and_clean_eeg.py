import mne
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler

# Define the file path
bdf_file = r"../data/EEG/P001.bdf"

# Load the raw BDF file
raw = mne.io.read_raw_bdf(bdf_file, preload=True)

print(raw.info['ch_names'])

# EarL, EXG2, EXG3, EXG4, EXG5, EXG6, EXG7, EXG8, GSR1, GSR2, Erg1, Erg2, Resp, Plet, Temp
raw = raw.set_montage("biosemi32", on_missing='ignore')

# picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=True, eog=False)
# print(picks)
# raw = raw.pick(picks)


raw = raw.drop_channels(["EarL", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8",
                         "GSR1", "GSR2", "Erg1", "Erg2", "Resp", "Plet", "Temp"])

# Set average reference
raw = raw.set_eeg_reference("average")

# Resample frequency
raw = raw.resample(512)

# Apply filters
raw = raw.filter(l_freq=1, h_freq=200)

# raw = raw.notch_filter(freqs=[50, 100], method='spectrum_fit')

mne.export.export_raw(r"../data/EEG/P001.set", raw, fmt='eeglab', overwrite=True)

mne.viz.plot_raw_psd(raw, fmax=100)

plt.show()
