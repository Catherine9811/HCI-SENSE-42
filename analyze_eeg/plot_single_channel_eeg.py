import mne
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler

# Define the file path
bdf_file = r"../data/EEG/P002.bdf"

# Load the raw BDF file
raw = mne.io.read_raw_bdf(bdf_file, preload=True)

print(raw.info['ch_names'])

# EarL, EXG2, EXG3, EXG4, EXG5, EXG6, EXG7, EXG8, GSR1, GSR2, Erg1, Erg2, Resp, Plet, Temp
raw = raw.set_montage("biosemi32", on_missing='ignore')

raw = raw.pick(['EarL', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2'])

# Resample frequency
raw = raw.resample(512)

# Apply filters
raw = raw.filter(l_freq=1, h_freq=200)

mne.viz.plot_raw(raw)

mne.viz.plot_raw_psd(raw, fmax=100)

plt.show()
