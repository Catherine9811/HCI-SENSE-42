import mne
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler

# Define the file path
participant_id = "P007"
fif_file = rf"../data/ECG/{participant_id}.fif"

raw_ecg = mne.io.read_raw_fif(fif_file, preload=True)
# Detect ECG events (R-waves) in lead II using find_ecg_events
# Note: find_ecg_events returns three values; here we only need the first one which contains event positions.
ecg_events, ch_ecg, average_pulse = mne.preprocessing.find_ecg_events(raw_ecg, ch_name='ECG2')

# Plot the derived 3-lead ECG signals
# raw_ecg.plot(title='Derived 3-lead ECG signals', events=ecg_events, duration=10, n_channels=3)

# Compute event times in seconds
sfreq = raw_ecg.info['sfreq']
event_times = ecg_events[:, 0] / sfreq

# Compute R-R intervals (the time difference between consecutive R-waves)
rr_intervals = np.diff(event_times)
# Use the midpoint of each interval for the time axis
rr_time = (event_times[:-1] + event_times[1:]) / 2

# Clean the data
min_bpm = 50
max_bpm = 220
valid_intervals = np.logical_and(rr_intervals >= 60.0 / max_bpm, rr_intervals <= 60.0 / min_bpm)
print(f"{len(valid_intervals)} samples found")
rr_intervals = rr_intervals[valid_intervals]
rr_time = rr_time[valid_intervals]

# Plot R-R intervals vs. time
plt.figure(figsize=(10, 5))
plt.scatter(rr_time, rr_intervals, marker='o', alpha=0.2, linestyle='-')
plt.xlabel('Time (seconds)')
plt.ylabel('R-R interval (seconds)')
plt.title('R-R Interval Scatter vs. Time')
plt.grid(True)
plt.show()


# Define the analysis window size in minutes and convert to seconds
window_size = 10  # minutes
window_sec = window_size * 60

# Get the total duration of the measurement in seconds
duration = raw_ecg.times[-1]

# Create time bins for every 10 minutes
time_bins = np.arange(0, duration, window_sec)

# Initialize lists to store the average R-R interval and its standard deviation (HRV) per window
avg_rr_intervals = []
std_rr_intervals = []

# For each time window, calculate the mean and standard deviation of the R-R intervals
for start in time_bins:
    end = start + window_sec
    # Get indices of R-R intervals falling into the current window
    indices = np.where((rr_time >= start) & (rr_time < end))[0]
    if len(indices) > 0:
        rr_window = rr_intervals[indices]
        avg_rr_intervals.append(np.mean(rr_window))
        std_rr_intervals.append(np.std(rr_window))
    else:
        avg_rr_intervals.append(np.nan)
        std_rr_intervals.append(np.nan)

# Compute the center time of each bin for plotting purposes
bin_centers = time_bins + window_sec / 2

# Plot the average R-R interval with error bars showing the standard deviation
fig, ax = plt.subplots(1, 1)

ax.errorbar(bin_centers, avg_rr_intervals, yerr=std_rr_intervals,
            fmt='o-', capsize=5, label="R-R Interval")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("R-R Interval (seconds)")
ax.set_title("Average and Variability of R-R Interval over Time")
ax.grid(True)

plt.tight_layout()
plt.show()

