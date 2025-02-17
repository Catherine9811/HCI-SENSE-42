import mne
import matplotlib.pyplot as plt
import addcopyfighandler

# Replace 'your_file.set' with the path to your .set file
raw = mne.io.read_raw_eeglab(r"../data/EEG/P001_RM_PowerLine_ICA_Muscle.set", preload=True)

# Total duration in seconds
total_duration = raw.times[-1]
total_duration_minutes = total_duration / 60
print(f"Total recording duration: {total_duration_minutes:.2f} minutes")

# Duration of each segment in seconds
segment_duration = 30 * 60  # 30 minutes

# Extract the first 30 minutes
first_segment = raw.copy().crop(tmin=0, tmax=segment_duration)

# Extract the last 30 minutes
tmin_last_segment = total_duration - segment_duration
last_segment = raw.copy().crop(tmin=tmin_last_segment, tmax=total_duration)

# Plot PSD for the first 30 minutes
fig, ax = plt.subplots(1, 2)
first_segment.plot_psd(ax=ax[0], fmax=20, average=True, show=False)
ax[0].set_title('PSD of First 30 Minutes')
ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Power Spectral Density (dB)')
ax[0].set_ylim(-20, 20)
# Plot PSD for the last 30 minutes
last_segment.plot_psd(ax=ax[1], fmax=20, average=True, show=False)
ax[1].set_title('PSD of Last 30 Minutes')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Power Spectral Density (dB)')
ax[1].set_ylim(-20, 20)
plt.show()