import numpy as np
import pickle
import jellyfish
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from data_parser import DataParser


def get_final_text(keys, rt, duration, ignore_deletion=False):
    text = ""
    shift_active = set()

    events = sorted(zip(rt, keys, duration), key=lambda x: x[0])

    for time, key, dur in events:
        if key in ["lshift", "rshift"]:
            shift_active.add((time, time + dur))  # Store shift release time

    for time, key, dur in events:
        if key in ["lshift", "rshift", "return"]:
            continue
        elif key == "backspace":
            if not ignore_deletion:
                text = text[:-1]  # Remove last character
        elif key == "space":
            text += " "
        else:
            # Check if any shift key is active at the moment
            if any(time <= time_pressed[1] and time >= time_pressed[0] for time_pressed in shift_active):
                text += key.upper()
            else:
                text += key.lower()

    return text

file_path = r"../data/001_explorer_2025-02-15_15h23.13.921.psydat"
parser = DataParser(file_path)
print(parser)

# Create the plot
plt.figure(figsize=(6, 4))
# Apply paper-style settings
sns.set_theme(style="whitegrid", context="paper")

for task_name, task_key, task_prefix, task_keyboard in [
    ('Shadow Typing', 'mail_content', 'single_note', 'mail.mail_content_user_key_release'),
    ('Side-by-side Typing', 'notes_repeat', 'notes', 'notes.notes_repeat_keyboard')
]:
    typing_task = parser[task_key]
    # Extracting values for plotting
    x_values = [entry[f"{task_key}.started"] for entry in typing_task]
    y_values = []
    for entry in typing_task:
        text = get_final_text(entry[f"{task_keyboard}.keys"], entry[f"{task_keyboard}.rt"], entry[f"{task_keyboard}.duration"], ignore_deletion=True)
        y_values.append(jellyfish.jaro_similarity(entry[f"{task_prefix}_repeat_source"], text))
    # Enable when the data collection is fixed
    # y_values = [jellyfish.jaro_similarity(entry[f"{task_prefix}_repeat_source"], entry[f"{task_prefix}_repeat_target"]) for entry in typing_task]

    plt.plot(x_values, y_values, marker='o', linestyle='-', label=task_name)

# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Jaro–Winkler Similarity (%)")
plt.title("Editing Distance in Typing Tasks Over Time")
plt.legend()

# Show the plot
plt.show()

