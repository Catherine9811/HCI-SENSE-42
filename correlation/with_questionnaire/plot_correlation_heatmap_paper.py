# --- add imports at top ---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

import addcopyfighandler
# ---- User parameters ----
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

outcome_variable = "sleepiness"
input_file = f"processed_data/42-{outcome_variable}-multimodal.csv"
output_file = f"processed_data/42-{outcome_variable}-multimodal-correlation-output.csv"

# ---- Load data ----

data = pd.read_csv(input_file)
data[outcome_variable] = pd.to_numeric(data[outcome_variable], errors='coerce')

# ---- Variables to keep ----

keep_vars = {
    "respiratory_inhalation_duration_mean": "Inhal. Time Mean",
    "respiratory_inhalation_duration_var": "Inhal. Time SD",
    "respiratory_exhalation_duration_mean": "Exhal. Time Mean",
    "respiratory_exhalation_duration_var": "Exhal. Time SD",
    "cardiac_rr_interval_mean": "R-R Interval",
    "cardiac_rr_interval_var": "Heart Rate Variability",
    "alpha": "Alpha Band Power",
    "keyboard_pressed_duration_mean": "Key-down Time Mean",
    "keyboard_pressed_duration_var": "Key-down Time SD",
    "keyboard_shadow_typing_efficiency_mean": "Typing Acc. Mean",
    "keyboard_typing_speed_mean": "Typing Speed Mean",
    "mouse_double_click_distance_mean": "Click Error Mean",
    "mouse_drag_distance_mean": "Drag Error Mean",
    "mouse_drag_distance_var": "Drag Error SD",
    "mouse_drop_distance_mean": "Drop Error Mean",
    "mouse_drop_distance_var": "Drop Error SD",
    "head_pitch_variation_mean": "Pitch Angle Variation",
    "head_pose_variation_mean": "Pose Variation",
    "blink_times_mean": "Eye Blink Count",
    # "look_down_times_mean": "Look Down Count",
    outcome_variable: "Sleepiness Score"
}

data = data[[col for col in keep_vars if col in data.columns]]

# ---- Compute correlation ----
data = data.rename(columns=lambda x: keep_vars[x])
corr_matrix = data.corr(method='pearson')

# --- add imports at top ---
import numpy as np
import textwrap
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---- Plot heatmap (triangular + diagonal label boxes, no manual grouping) ----
sns.set(style="white")
# wider figure => wider cells; also set square=False to allow rectangular cells
fig, ax = plt.subplots(figsize=(18, 10))

# 1) Hierarchical clustering on all variables (distance = 1 - corr)
plot_corr = corr_matrix.copy().astype(float)

# 2) Show only LOWER triangle; keep diagonal
mask = np.triu(np.ones(plot_corr.shape, dtype=bool), k=1)

# 3) Annotation: numbers only in lower triangle
vals = plot_corr.values
annot = np.empty_like(vals, dtype=object)
n = vals.shape[0]
for i in range(n):
    for j in range(n):
        annot[i, j] = f"{vals[i, j]:.2f}" if i > j else ""

# 4) Heatmap with short colorbar and clean blank half (no grids there)
cmap = mpl.cm.get_cmap("coolwarm").copy()
cmap.set_bad("white")

hm = sns.heatmap(
    plot_corr,
    mask=mask,
    cmap=cmap,
    vmin=-1, vmax=1, center=0,
    square=False,                    # allow horizontal expansion
    linewidths=2.0, linecolor="white",
    annot=annot, fmt="",
    annot_kws={"size": 10},
    cbar=False,
    cbar_kws={"shrink": 0.60, "pad": 0.02, "aspect": 30},
    ax=ax
)

# Horizontal colorbar at top-right (inside the axes)
cax = inset_axes(
    ax,
    width="28%",   # tweak size
    height="3%",   # tweak thickness
    loc="upper right",
    borderpad=0.8  # tweak padding from corner
)

mappable = ax.collections[0]
cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
cb.set_ticks([-1, -0.5, 0, 0.5, 1])

# move tick labels to bottom
cb.ax.xaxis.set_ticks_position("bottom")
cb.ax.xaxis.set_label_position("bottom")

# remove the small tick marks on the bar (keep labels)
# turn off minor ticks completely
# cb.ax.minorticks_off()
#
# # remove tick marks (keep labels)
# cb.ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=True, length=0, width=0)
#
# # hard-disable any remaining tick lines
# for t in cb.ax.xaxis.get_major_ticks():
#     t.tick1line.set_visible(False)
#     t.tick2line.set_visible(False)
#
# # if you also want no border for the colorbar:
# cb.outline.set_visible(False)
# for spine in cb.ax.spines.values():
#     spine.set_visible(False)

# 5) Diagonal label boxes: grey background, same size as cells (1x1 in heatmap coords)
diag_face = "#e6e6e6"
for i, name in enumerate(plot_corr.columns):
    ax.add_patch(Rectangle((i, i), 1, 1, facecolor=diag_face, edgecolor="white", linewidth=2))
    ax.text(
        i + 0.5, i + 0.5,
        textwrap.fill(name, width=12),  # increase width to reduce wrapping
        ha="center", va="center",
        fontsize=8, color="#222222"
    )


# ✅ 6) Modality annotations to the right of diagonal (skip outcome_variable)
def modality_from_orig(orig_name: str) -> str:
    if orig_name.startswith("respiratory_"):
        return "Physiological\n(Respiratory)"
    if orig_name.startswith("cardiac_"):
        return "Physiological\n(ECG)"
    if orig_name == "alpha" or orig_name.startswith("eeg_"):
        return "Physiological\n(EEG)"
    if orig_name.startswith("keyboard_"):
        return "Behavioural\n(Keyboard)"
    if orig_name.startswith("mouse_"):
        return "Behavioural\n(Mouse)"
    if orig_name.startswith("head_"):
        return "Behavioural\n(Webcam)"
    if orig_name.startswith("blink_"):
        return "Behavioural\n(Webcam)"
    return ""


pretty_to_orig = {keep_vars[k]: k for k in keep_vars}  # ✅ 反查：显示名 -> 原始名

pretty_cols = list(plot_corr.columns)
n = len(pretty_cols)
# 给右侧留出空间
ax.set_xlim(0, n+0.5)
# 按“相邻连续列”的模态分组（并跳过 outcome_variable）
groups = []
i = 0
while i < n:
    pretty = pretty_cols[i]
    orig = pretty_to_orig.get(pretty, "")
    if orig == outcome_variable:
        i += 1
        continue

    m = modality_from_orig(orig)
    s = i
    j = i
    while j + 1 < n:
        pretty2 = pretty_cols[j + 1]
        orig2 = pretty_to_orig.get(pretty2, "")
        if orig2 == outcome_variable or modality_from_orig(orig2) != m:
            break
        j += 1
    e = j
    groups.append((m, s, e))
    i = e + 1

annot_face = "#888"
# 画：从每个对角线格子右边缘出发的水平线 + 右侧垂直线 + 模态文字
for m, s, e in groups:
    x_br = (e + 1) + 0.35  # 垂直线的 x：放在该模态最后一个对角格子的右侧
    # 水平“引线”：每个对角格子 -> 垂直线
    for idx in range(s, e + 1):
        if idx != s and idx != e and s != e:
            continue
        y = idx + 0.5
        ax.plot([idx + 1.08, x_br], [y, y], color=annot_face, lw=1.2, clip_on=False)
    # 垂直线：覆盖该模态 span
    if s == e:
        ax.plot([x_br, x_br], [s + 0.10, e + 0.90], color=annot_face, lw=1.2, clip_on=False)
    else:
        ax.plot([x_br, x_br], [s + 0.50, e + 0.50], color=annot_face, lw=1.2, clip_on=False)
    # 文字：垂直线右侧居中
    ax.text(x_br + 0.12, (s + e + 1) / 2, m, ha="left", va="center",
            fontsize=11, color=annot_face, clip_on=False)


# Hide axis tick labels (diagonal boxes carry labels)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(length=0)

# Remove outer borders (plot + colorbar)
for spine in ax.spines.values():
    spine.set_visible(False)

cb = ax.collections[0].colorbar
cb.outline.set_visible(False)
for spine in cb.ax.spines.values():
    spine.set_visible(False)

cb = ax.collections[0].colorbar
cb.outline.set_edgecolor("black")
cb.outline.set_linewidth(1.2)
cb.ax.tick_params(length=3, width=1.0, colors="black")

# ax.set_title(f"Correlation Heatmap for {outcome_variable.title()}", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
plt.savefig(f"processed_data/42-{outcome_variable}-correlation-heatmap.png", dpi=300, bbox_inches="tight")
plt.show()