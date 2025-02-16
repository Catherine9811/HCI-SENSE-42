import numpy as np
import pickle
import jellyfish
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from data_parser import DataParser


def count_effective_keys(keys):
    return len(keys) - 2 * keys.count('backspace')


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
    y_values = [
        count_effective_keys(entry[f"{task_keyboard}.keys"]) / (entry[f"{task_key}.stopped"] - entry[f"{task_key}.started"])
        for entry in typing_task]

    plt.plot(x_values, y_values, marker='o', linestyle='-', label=task_name)

# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Typed Characters per Second")
plt.title("Typing Speed in Typing Tasks Over Time")
plt.legend()

# Show the plot
plt.show()

