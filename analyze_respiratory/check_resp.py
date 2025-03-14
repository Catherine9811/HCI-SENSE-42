from analyze_respiratory.rip import Resp
import datetime
import numpy as np
import addcopyfighandler
import matplotlib.pyplot as plt


def round_by(seconds, units):
    return int(seconds/units) * units


resp = Resp.from_wav(r"../data/Respiration/P006.wav")

print('''Sampling frequency: {}
Number of samples: {}
Duration: {}'''.format(resp.samp_freq, len(resp.samples), datetime.timedelta(seconds=resp.dur)))

baseline = resp.baseline_savgol(60)
# plt.plot(resp.t, resp.samples)
# plt.plot(resp.t, baseline)

resp.remove_baseline(method='savgol')
# plt.plot(resp.t, resp.samples)

resp.find_cycles(include_holds=True)

print(resp.segments)

visualize_slice = slice(0, 5120)

plt.plot(resp.t[visualize_slice], resp.samples[visualize_slice], color='lightgrey')
plt.plot(resp.peaks[visualize_slice], resp.idt[resp.peaks[visualize_slice]],
         linestyle='none', marker='o')
plt.plot(resp.troughs[visualize_slice], resp.idt[resp.troughs[visualize_slice]],
         linestyle='none', marker='o')
plt.xlim(0, 5120 / resp.samp_freq)
plt.show()

inh_durs = [i.duration() for i in resp.inhalations]

exh_durs = [i.duration() for i in resp.exhalations]

plt.hist(inh_durs, bins=20)
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.title("Inhalation Distribution")
plt.show()

plt.hist(exh_durs, bins=20)
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.title("Exhalation Distribution")
plt.show()

window_size = 10    # minutes
time_bins = np.arange(0, round_by(resp.dur, 60 * window_size), 60 * window_size)  # Bins for every 10 minutes
avg_inh_durs = []
avg_exh_durs = []
std_inh_durs = []
std_exh_durs = []
breathing_rates = []

for start in time_bins:
    end = start + 60 * window_size
    inh_durs_window = [i.duration() for i in resp.inhalations if start <= i.start_time < end]
    exh_durs_window = [i.duration() for i in resp.exhalations if start <= i.start_time < end]
    breath_count = (len(inh_durs_window) + len(exh_durs_window)) / 2
    breathing_rates.append(breath_count / window_size)  # Breaths per minute

    avg_inh_durs.append(np.mean(inh_durs_window) if inh_durs_window else 0)
    avg_exh_durs.append(np.mean(exh_durs_window) if exh_durs_window else 0)
    std_inh_durs.append(np.std(inh_durs_window) if inh_durs_window else 0)
    std_exh_durs.append(np.std(exh_durs_window) if exh_durs_window else 0)

fig, ax = plt.subplots(2, 1)
ax[0].errorbar(time_bins, avg_inh_durs, yerr=std_inh_durs, fmt='o', label="Inhalation Duration", capsize=5)
ax[1].errorbar(time_bins, avg_exh_durs, yerr=std_exh_durs, fmt='s', label="Exhalation Duration", capsize=5)
# ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Inhalation Duration (seconds)")
ax[1].set_xlabel("Time (seconds)")
ax[1].set_ylabel("Exhalation Duration (seconds)")
ax[0].set_title("Average Inhalation and Exhalation Durations with Error Bars")

ax[0].grid()
ax[1].grid()
plt.show()

# Plot breathing rate over time
plt.figure(figsize=(12, 5))
plt.plot(time_bins, breathing_rates, marker='o', linestyle='-', label="Breathing Rate (breaths per minute)")
plt.xlabel("Time (seconds)")
plt.ylabel("Breathing Rate (breaths per minute)")
plt.title("Average Breathing Rate Over Time")
plt.legend()
plt.grid()
plt.show()