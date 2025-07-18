import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from data_parser import DataParser


file_path = r"data\001_explorer_2025-02-15_15h23.13.921.psydat"
parser = DataParser(file_path)
print(parser)

"""
Questionnaire Keys
sleepiness:_how_sleepy_are_you_slider
mental_demand:_how_mentally_demanding_was_the_task_slider
physical_demand:_how_physically_demanding_was_the_task_slider
temporal_demand:_how_hurried_or_rushed_was_the_pace_of_the_task_slider
performance:_how_successful_were_you_in_accomplishing_what_you_were_asked_to_do_slider
effort:_how_hard_did_you_have_to_work_to_accomplish_your_level_of_performance_slider
frustration:_how_insecure__discouraged__irritated__stressed__and_annoyed_were_you_slider
"""

questionnaires = parser['browser_content']

# Extracting values for plotting
x_values = [entry["browser_content.started"] for entry in questionnaires]
y_values = [entry["sleepiness:_how_sleepy_are_you_slider"] for entry in questionnaires]

# Apply paper-style settings
sns.set_theme(style="whitegrid", context="paper")

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label="Sleepiness Level")

# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Sleepiness Level (1-9)")
plt.title("Sleepiness Level Over Time")
plt.legend()

# Show the plot
plt.show()

