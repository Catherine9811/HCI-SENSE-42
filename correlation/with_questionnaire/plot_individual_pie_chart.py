import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import addcopyfighandler


# Set figure style for paper
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'savefig.dpi': 300,
})

sns.set_theme(style="whitegrid", context="paper")


# ---- Load CSV ----

input_file = "processed_data/42-sleepiness-multimodal-individual-contributions.csv"  # replace with your actual path
data = pd.read_csv(input_file)

# ---- Pie chart parameters ----
color_dict = {
    "Physiological (EEG,ECG)": "#1F78B4",
    "Physiological (Respiration)": "#33A1C9",
    "Behavioural (Webcam)": "#5CC8FF",
    "Behavioural (Mouse)": "#D55E00",
    "Behavioural (Keyboard)": "#E69F00",
    "Individual Traits": "#98df8a",
    "Quality of Sleep": "#006400",
    "Fixed Effect Shared Contribution": "#999999",
    "Random Effect Contribution": "#D9D9D9",
    "Unexplained Variance": "#FFFFFF"
}
labels = data["term"]
sizes = data["value"]  # or data["pct"] if you prefer
colors = [color_dict[label] for label in labels]
explode = [0.01 if pct < 0.1 else 0 for pct in data["pct"]]  # slight offset for small slices

# ---- Plot ----

plt.figure(figsize=(8, 8))
patches, texts, autotexts = plt.pie(
    sizes,
    labels=None,  # hide default labels
    autopct=lambda p: '{:.0f}%'.format(p) if p > 1 else '',  # annotate only if >1%
    startangle=0,
    colors=colors[:len(sizes)],
    explode=explode,
    wedgeprops={"edgecolor": "black", "linewidth": 1.0},
    pctdistance=1.1
)

# ---- Customize text ----

# for autotext in autotexts:
#     autotext.set_color('white')
#     autotext.set_fontsize(10)
#     autotext.set_weight('bold')

# plt.title("Breakdown for Explained Variance of Sleepiness (KSS)", fontsize=12)

# ---- Add legend ----

legend = plt.legend(patches, labels, title="Predictor Sensor Modality",
                    bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, frameon=False)
legend._legend_box.align = "left"
legend.get_title().set_ha("left")

plt.tight_layout()
plt.savefig("processed_data/sleepiness_individual_variance_pie.png", dpi=300)
plt.show()
