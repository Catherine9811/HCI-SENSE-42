import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from data_parser import DataParser


file_path = r"../data/006_explorer_2025-03-13_19h45.43.154.psydat"
parser = DataParser(file_path)
print(parser)

"""
Questionnaire Keys
sleepiness:_how_sleepy_are_you_slider
mental_demand:_how_mentally_demanding_was_the_task_slider
#physical_demand:_how_physically_demanding_was_the_task_slider
temporal_demand:_how_hurried_or_rushed_was_the_pace_of_the_task_slider
performance:_how_successful_were_you_in_accomplishing_what_you_were_asked_to_do_slider
effort:_how_hard_did_you_have_to_work_to_accomplish_your_level_of_performance_slider
frustration:_how_insecure__discouraged__irritated__stressed__and_annoyed_were_you_slider
attentiveness:_how_focused_were_you_on_performing_the_task_slider
"""

questionnaires = parser['browser_content']

questions = [
    "sleepiness:_how_sleepy_are_you_slider",
    "mental_demand:_how_mentally_demanding_was_the_task_slider",
    "temporal_demand:_how_hurried_or_rushed_was_the_pace_of_the_task_slider",
    "performance:_how_successful_were_you_in_accomplishing_what_you_were_asked_to_do_slider",
    "effort:_how_hard_did_you_have_to_work_to_accomplish_your_level_of_performance_slider",
    "frustration:_how_insecure__discouraged__irritated__stressed__and_annoyed_were_you_slider",
    "attentiveness:_how_focused_were_you_on_performing_the_task_slider"
]

titles = [
    "Sleepiness",
    "Mental Demand",
    "Temporal Demand",
    "Performance",
    "Effort",
    "Frustration",
    "Attentiveness"
]

ranges = [
    "(1-9)",
    "(1-7)",
    "(1-7)",
    "(1-7)",
    "(1-7)",
    "(1-7)",
    "(1-7)"
]

# Apply paper-style settings
sns.set_theme(style="whitegrid", context="paper")

# Create the plot
plt.figure(figsize=(6, 4))

# Extracting values for plotting
for question_name, question_title, question_range in zip(questions, titles, ranges):
    x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
    y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]

    plt.plot(x_values, y_values, marker='o', linestyle='-', label=f"{question_title} {question_range}")

# Labels and title
plt.xlabel("Time (seconds)")
# plt.ylabel("Sleepiness Level (1-9)")
plt.ylabel("Self-reported Values")
# plt.title("Sleepiness Level Over Time")
plt.title("Questionnaire Reported Values Over Time")
plt.legend()

# Show the plot
plt.show()

