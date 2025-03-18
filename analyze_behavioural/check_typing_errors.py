import numpy as np
import pickle
import jellyfish
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from data_parser import DataParser


file_path = r"../data/007_explorer_2025-03-14_14h12.43.192.psydat"
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
        y_values.append(entry[f"{task_keyboard}.keys"].count("backspace"))

    plt.plot(x_values, y_values, marker='o', linestyle='-', label=task_name)

# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Number of Deletions")
plt.title("Errors in Typing Tasks Over Time")
plt.legend()

# Show the plot
plt.show()

