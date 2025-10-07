import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import addcopyfighandler
import seaborn as sns

# === Settings ===
saving_psd_dir = os.path.join("data", "psd_welch_event7to9")
frequency_bands = {
    "delta": [1.0, 4.0],
    "theta": [4.0, 8.0],
    "alpha": [8.0, 14.0],
    "beta": [14.0, 30.0],
    "gamma": [30.0, 100.0],
}
eeg_bands = ["delta", "theta", "alpha", "beta", "gamma"]

# Find all PSD files
psd_files = sorted(glob(os.path.join(saving_psd_dir, "P*-psd-event7to9-*.npz")))

# Collect data for plotting
plot_data = []

for psd_file in psd_files:
    fname = os.path.basename(psd_file)
    participant_id = int(fname[1:4])
    if participant_id not in [1, 2, 6, 8, 19, 25, 26, 29, 30, 31, 33, 34, 39, 40, 42]:
        continue
    num_events = int(fname.split("-")[-1].split(".")[0])

    data = np.load(psd_file)
    psds = data["psds"]
    freqs = data["freqs"].squeeze()
    mean_psd = psds.mean(axis=0)

    for band in eeg_bands:
        fmin, fmax = frequency_bands[band]
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if len(idx) > 0:
            power = mean_psd[idx].mean()
        else:
            power = np.nan
        plot_data.append({
            "participant_id": participant_id,
            "num_events": num_events,
            "band": band,
            "power": 10 * np.log10(power) + 120
        })

# Convert to DataFrame
df_plot = pd.DataFrame(plot_data)

# --- Remove outliers per participant per band ---
def remove_outliers(group):
    q1 = group["power"].quantile(0.25)
    q3 = group["power"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return group[(group["power"] >= lower) & (group["power"] <= upper)]

df_plot = df_plot.groupby(["participant_id", "band"], group_keys=False).apply(remove_outliers)

# --- Apply heavy running average per participant ---
def apply_smoothing(group):
    group = group.sort_values("num_events")
    group["power_smooth"] = group["power"].rolling(
        window=10, min_periods=1, center=True
    ).mean()
    return group

df_plot = df_plot.groupby(["participant_id", "band"], group_keys=False).apply(apply_smoothing)

# --- Plotting ---
sns.set(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("tab20", n_colors=df_plot["participant_id"].nunique())

for band in eeg_bands:
    plt.figure(figsize=(8, 5))
    band_data = df_plot[df_plot["band"] == band]
    sns.scatterplot(
        data=band_data,
        x="num_events",
        y="power_smooth",
        hue="participant_id",
        palette=palette,
        alpha=0.8,
        legend="full"
    )
    # Optionally connect points per participant
    for pindex, pid in enumerate(band_data["participant_id"].unique()):
        participant_data = band_data[band_data["participant_id"] == pid].sort_values("num_events")
        plt.plot(
            participant_data["num_events"],
            participant_data["power_smooth"],
            color=palette[pindex],
            alpha=0.8
        )

    plt.title(f"{band.capitalize()} band power vs. num_events")
    plt.xlabel("Event pair index (num_events)")
    plt.ylabel("Power")
    plt.legend(title="Participant", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
