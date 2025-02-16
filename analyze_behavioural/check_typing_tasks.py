import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from data_parser import DataParser

file_path = r"../data/001_explorer_2025-02-15_15h23.13.921.psydat"
parser = DataParser(file_path)
print(parser)

# Create the plot
plt.figure(figsize=(6, 4))
# Apply paper-style settings
sns.set_theme(style="whitegrid", context="paper")

for task_name, task_key in [
    ('Shadow Typing', 'mail_content'),
    ('Side-by-side Typing', 'notes_repeat')
]:
    typing_task = parser[task_key]

    # Extracting values for plotting
    x_values = [entry[f"{task_key}.started"] for entry in typing_task]
    y_values = [entry[f"{task_key}.stopped"] - entry[f"{task_key}.started"] for entry in typing_task]

    plt.plot(x_values, y_values, marker='o', linestyle='-', label=task_name)

# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Time Spent (seconds)")
plt.title("Time Used in Typing Tasks Over Time")
plt.legend()

# Show the plot
plt.show()

