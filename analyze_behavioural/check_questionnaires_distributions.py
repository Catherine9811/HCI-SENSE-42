import numpy as np
import os
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler
from tqdm import tqdm

from data_parser import DataParser
from data_definition import psydat_files

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12


questions = [
    "attentiveness:_how_focused_were_you_on_performing_the_task_slider",
    "mental_demand:_how_mentally_demanding_was_the_task_slider",
    "temporal_demand:_how_hurried_or_rushed_was_the_pace_of_the_task_slider",
    "performance:_how_successful_were_you_in_accomplishing_what_you_were_asked_to_do_slider",
    "effort:_how_hard_did_you_have_to_work_to_accomplish_your_level_of_performance_slider",
    "frustration:_how_insecure__discouraged__irritated__stressed__and_annoyed_were_you_slider",
    "sleepiness:_how_sleepy_are_you_slider",
]

titles = [
    "Attentiveness",
    "Mental Demand",
    "Temporal Demand",
    "Performance",
    "Effort",
    "Frustration",
    "Sleepiness",
]

ranges = [
    (1, 7),
    (1, 7),
    (1, 7),
    (1, 7),
    (1, 7),
    (1, 7),
    (1, 9)
]

labels = [
    ["not at all", "", "", "", "", "", "completely"],
    ["very low", "", "", "medium", "", "", "very high"],
    ["very slow", "", "", "medium", "", "", "very fast"],
    ["failure", "", "", "okay", "", "", "perfect"],
    ["very low", "", "", "medium", "", "", "very high"],
    ["very low", "", "", "medium", "", "", "very high"],
    ["extremely alert", "", "", "", "neutral", "", "", "", "very sleepy"],
]

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

cache_file = os.path.join(CACHE_DIR, "aggregated_data.pkl")

if os.path.exists(cache_file):
    # Load cached data
    with open(cache_file, "rb") as f:
        aggregated_data, aggregated_gap = pickle.load(f)
    print("Loaded cached data from", cache_file)
else:
    # Compute and cache data
    aggregated_data = {title: [] for title in titles}
    aggregated_gap = {title: [] for title in titles}

    for file_path in tqdm(psydat_files):
        parser = DataParser(os.path.join("..", "data", file_path))
        paused_blocks = parser["pause_on"]
        questionnaires = parser['browser_content']

        for question_name, question_title, question_range in zip(questions, titles, ranges):
            time_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"]
                           for entry in questionnaires]
            response_values = [entry[f"{question_name}.rating"] for entry in questionnaires]

            for index in range(len(time_values)):
                for block in paused_blocks:
                    if time_values[index] > block[f"pause_on.started"]:
                        time_values[index] -= block[f"pause_on.stopped"] - block[f"pause_on.started"]

            aggregated_gap[question_title].extend(np.diff(time_values))
            aggregated_data[question_title].extend(response_values)

    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump((aggregated_data, aggregated_gap), f)
    print("Saved processed data to cache:", cache_file)


overall_latency = []
for title in titles:
    print(title, "min:", np.min(aggregated_data[title]), "mean:", np.mean(aggregated_data[title]), "max:", np.max(aggregated_data[title]), "std", np.std(aggregated_data[title]), "distinct", np.unique(aggregated_data[title])) # / np.sqrt(len(psydat_files) - 1))
    print(f"{title} Gaps", "min:", np.min(aggregated_gap[title]), "mean:", np.mean(aggregated_gap[title]), "max:", np.max(aggregated_gap[title]), "std", np.std(aggregated_gap[title])) # / np.sqrt(len(psydat_files) - 1))
    overall_latency.extend(aggregated_gap[title])

print("Overall", np.mean(overall_latency), np.std(overall_latency))
# fig, axes = plt.subplots(2, len(titles), figsize=(14, 6))
#
# # Plot histograms
# for i, title in enumerate(titles):
#     # Left: raw variable
#     axes[0, i].hist(aggregated_data[title], bins=ranges[i][1], range=(ranges[i][0], ranges[i][1] + 1), color='tab:blue', alpha=0.7, edgecolor="steelblue")
#     # axes[0, i].set_title(f'{title}')
#     axes[0, i].set_xlabel(title)
#     if i == 0:
#         axes[0, i].set_ylabel('Frequency')
#     axes[0, i].set_ylim(0, 400)
#     axes[0, i].set_xticks(ticks=[i+0.5 for i in list(range(ranges[i][0], ranges[i][1] + 1))],
#                           labels=[str(i) for i in list(range(ranges[i][0], ranges[i][1] + 1))])
#     axes[0, i].grid(True, linestyle='--', alpha=0.5)
#     axes[0, i].spines['top'].set_visible(False)
#     axes[0, i].spines['right'].set_visible(False)
#     axes[0, i].grid(axis='x', visible=False)
#
#     # Right: gap
#     axes[1, i].hist(aggregated_gap[title], bins=30, color='darkorange', alpha=0.7)
#     axes[1, i].set_title(f'{title} Gaps')
#     axes[1, i].set_xlabel(f'{title} Gaps')
#     axes[1, i].set_ylabel('Frequency')
#     # axes[1, i].set_ylim(0, 40)
#     axes[1, i].grid(True, linestyle='--', alpha=0.5)
#     axes[1, i].spines['top'].set_visible(False)
#     axes[1, i].spines['right'].set_visible(False)
#     axes[1, i].grid(axis='x', visible=False)
#
# plt.show()

plt.figure(figsize=(20, 6))

x_offset = 0
bar_width = 0.8
gap_between_categories = 2  # space between variable histograms
x_tick_positions = []
x_tick_labels = []
max_height = 0

for i, title in enumerate(titles):
    values = aggregated_data[title]
    rating_range = ranges[i]
    bins = np.arange(rating_range[0], rating_range[1] + 2)  # inclusive upper edge
    counts, _ = np.histogram(values, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2 + x_offset

    plt.bar(bin_centers, counts, width=bar_width, alpha=0.7, color='tab:blue', edgecolor='black')

    # Add x-axis ticks for each rating bin
    for j, val in enumerate(range(rating_range[0], rating_range[1] + 1)):
        x_tick_positions.append(bin_centers[j])
        x_tick_labels.append(str(val))

    for j, val in enumerate(range(rating_range[0], rating_range[1] + 1)):
        plt.text(bin_centers[j], -20, labels[i][j], ha='center', va='top', fontsize=9)

    # Optional: Add variable label centered below its range
    mid_x = np.mean(bin_centers)
    plt.text(mid_x, -34, title, ha='center', va='top', fontsize=14)

    max_height = max(max_height, max(counts) if counts.size else 0)
    x_offset += (rating_range[1] - rating_range[0] + 1) + gap_between_categories

# Styling
plt.xticks(x_tick_positions, x_tick_labels, fontsize=9)
plt.ylabel('Frequency')
# plt.title('Distributions of Questionnaire Ratings')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.ylim(0, max_height * 1.05)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()