import mne
import matplotlib.pyplot as plt

# Define the file path
# bdf_file = r"../data/EEG/P038.bdf"
# bdf_file = r"../data/EEG/P005 02.bdf"
bdf_file = r"D:\HCI PROCESSED DATA\EEGRaw\P001.bdf"

# Load the raw BDF file
raw = mne.io.read_raw_bdf(bdf_file, preload=True)

print(raw.info['ch_names'])

# Extract events from stimulus channel
events = mne.find_events(raw, stim_channel='Status', initial_event=True, shortest_event=1)

# Create event_id mapping
event_id = {str(event): event for event in set(events[:, 2]) if event < 255}
# event_id = {str(event): event for event in set(events[:, 2])}

print(event_id)

temp_data, times = raw.copy().pick_channels(['Temp']).get_data(return_times=True)


# Plot the raw EEG data with events
raw.plot(events=events, scalings='auto', title='EEG Time Series with Events')

# Plot the events separately
mne.viz.plot_events(events, sfreq=raw.info['sfreq'], event_id=event_id)

plt.show()
