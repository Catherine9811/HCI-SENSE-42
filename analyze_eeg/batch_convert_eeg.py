import mne
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import addcopyfighandler

for participant_id in tqdm(range(1, 43)):
    # Define the file path
    bdf_file = rf"../data/EEG/P{participant_id:03d}.bdf"
    save_file = rf"../data/EEG_cleaned/P{participant_id:03d}.set"

    # Load the raw BDF file
    raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose="WARNING")

    if participant_id == 5:
        raw_extra = mne.io.read_raw_bdf(rf"../data/EEG/P{participant_id:03d} 02.bdf", preload=True)
        raw = mne.concatenate_raws([raw, raw_extra])

    raw = raw.drop_channels(["EarL", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8",
                             "GSR1", "GSR2", "Erg1", "Erg2", "Resp", "Plet", "Temp", "Status"])

    # EarL, EXG2, EXG3, EXG4, EXG5, EXG6, EXG7, EXG8, GSR1, GSR2, Erg1, Erg2, Resp, Plet, Temp
    raw = raw.set_montage("biosemi32", on_missing='ignore')

    # Resample frequency
    raw = raw.resample(512)

    mne.export.export_raw(save_file, raw, fmt='eeglab', overwrite=True)
