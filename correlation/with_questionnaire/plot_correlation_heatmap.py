import pandas as pd
import seaborn as sns
import matplotlib as mpl
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

# ---- User parameters ----

outcome_variable = "sleepiness"
input_file = f"processed_data/42-{outcome_variable}-multimodal.csv"
output_file = f"processed_data/42-{outcome_variable}-multimodal-correlation-output.csv"

# ---- Load data ----

data = pd.read_csv(input_file)
data[outcome_variable] = pd.to_numeric(data[outcome_variable], errors='coerce')

# ---- Variables to keep ----

keep_vars = [
    "respiratory_inhalation_duration_mean",
    "respiratory_inhalation_duration_var",
    "time",
    "mouse_drop_distance_mean",
    "keyboard_pressed_duration_mean",
    "cardiac_rr_interval_var",
    "respiratory_exhalation_duration_mean",
    "cardiac_rr_interval_mean",
    "alpha",
    "keyboard_pressed_duration_var",
    "respiratory_exhalation_duration_var",
    "mouse_grouped_selection_duration_var",
    "mouse_drag_distance_mean",
    "mouse_drop_distance_var",
    "head_pitch_variation_mean",
    "blink_times_mean",
    "mouse_drag_distance_var",
    "mouse_double_click_distance_mean",
    outcome_variable
]

data = data[[col for col in keep_vars if col in data.columns]]

# ---- Define categories ----

category_list = {
    "physiological_eeg": ["alpha", "beta", "theta", "delta", "gamma"],
    "physiological_ecg": ["cardiac_rr_interval_mean", "cardiac_rr_interval_var"],
    "physiological_resp": [
        "respiratory_inhalation_duration_mean", "respiratory_inhalation_duration_var",
        "respiratory_exhalation_duration_mean", "respiratory_exhalation_duration_var"
    ],
    "behavioural": [
        "head_pose_variation_mean", "head_pose_movement_mean", "head_pitch_variation_mean",
        "head_roll_variation_mean", "head_yaw_variation_mean", "blink_times_mean",
        "look_down_times_mean"
    ],
    "interaction_mouse": [
        "mouse_double_click_distance_mean",
        "mouse_double_click_distance_var",
        "mouse_drag_distance_mean",
        "mouse_drag_distance_var",
        "mouse_drop_distance_mean",
        "mouse_drop_distance_var",
        "mouse_taskbar_navigation_efficiency_mean",
        "mouse_taskbar_navigation_efficiency_var",
        "mouse_toolbar_navigation_efficiency_mean",
        "mouse_toolbar_navigation_efficiency_var",
        "mouse_selection_coverage_mean",
        "mouse_selection_coverage_var",
        "mouse_folder_navigation_speed_mean",
        "mouse_folder_navigation_speed_var",
        "mouse_toolbar_navigation_speed_mean",
        "mouse_toolbar_navigation_speed_var",
        "mouse_confirm_dialog_duration_mean",
        "mouse_confirm_dialog_duration_var",
        "mouse_notification_duration_mean",
        "mouse_notification_duration_var",
        "mouse_open_folder_duration_mean",
        "mouse_open_folder_duration_var",
        "mouse_drag_folder_duration_mean",
        "mouse_drag_folder_duration_var",
        "mouse_close_window_duration_mean",
        "mouse_close_window_duration_var",
        "mouse_grouped_selection_duration_mean",
        "mouse_grouped_selection_duration_var",
        "mouse_open_notes_duration_mean",
        "mouse_open_notes_duration_var",
        "mouse_open_browser_duration_mean",
        "mouse_open_browser_duration_var",
        "mouse_open_file_manager_duration_mean",
        "mouse_open_file_manager_duration_var",
        "mouse_open_trash_bin_duration_mean",
        "mouse_open_trash_bin_duration_var",
        "mouse_open_folder_clicking_duration_mean",
        "mouse_open_folder_clicking_duration_var",
        "mouse_close_window_clicking_duration_mean",
        "mouse_close_window_clicking_duration_var",
        "mouse_close_window_unintended_clicks_mean",
        "mouse_close_window_unintended_clicks_var",
        "mouse_open_folder_unintended_clicks_mean",
        "mouse_open_folder_unintended_clicks_var",
        "mouse_close_to_toolbar_navigation_speed_mean",
        "mouse_close_to_toolbar_navigation_speed_var",
        "mouse_confirm_dialog_unintended_clicks_mean",
        "mouse_confirm_dialog_unintended_clicks_var",
        "mouse_open_notes_unintended_clicks_mean",
        "mouse_open_notes_unintended_clicks_var",
        "mouse_open_browser_unintended_clicks_mean",
        "mouse_open_browser_unintended_clicks_var",
        "mouse_open_file_manager_unintended_clicks_mean",
        "mouse_open_file_manager_unintended_clicks_var",
        "mouse_open_trash_bin_unintended_clicks_mean",
        "mouse_open_trash_bin_unintended_clicks_var",
        "mouse_open_notification_unintended_clicks_mean",
        "mouse_open_notification_unintended_clicks_var"
    ],
    "interaction_keyboard": [
        "keyboard_shadow_typing_duration_mean",
        "keyboard_shadow_typing_duration_var",
        "keyboard_side_by_side_typing_duration_mean",
        "keyboard_side_by_side_typing_duration_var",
        "keyboard_shadow_typing_error_mean",
        "keyboard_shadow_typing_error_var",
        "keyboard_side_by_side_typing_error_mean",
        "keyboard_side_by_side_typing_error_var",
        "keyboard_typing_speed_mean",
        "keyboard_typing_speed_var",
        "keyboard_space_key_pressed_duration_mean",
        "keyboard_space_key_pressed_duration_var",
        "keyboard_space_key_typing_duration_mean",
        "keyboard_space_key_typing_duration_var",
        "keyboard_pressed_duration_mean",
        "keyboard_pressed_duration_var",
        "keyboard_shadow_typing_efficiency_mean",
        "keyboard_shadow_typing_efficiency_var",
        "keyboard_side_by_side_typing_efficiency_mean",
        "keyboard_side_by_side_typing_efficiency_var"
    ]
}

# ---- Compute correlation ----
data = data.rename(columns=lambda x: x.replace("_", " ").title().replace("Rr", "R-R"))
corr_matrix = data.corr(method='pearson')

# ---- Save correlation matrix ----

corr_matrix.to_csv(output_file, index=True)

# ---- Plot heatmap ----

plt.figure(figsize=(12, 10))
sns.set(style="white")

# Optional: reorder variables by category

ordered_vars = []
for cat, vars_in_cat in category_list.items():
    ordered_vars.extend([v.replace("_", " ").title().replace("Rr", "R-R") for v in vars_in_cat if v.replace("_", " ").title().replace("Rr", "R-R") in data.columns])
ordered_vars.append(outcome_variable.replace("_", " ").title().replace("Rr", "R-R"))  # outcome at the end

sns.heatmap(
    corr_matrix.loc[ordered_vars, ordered_vars],
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=1.0,
    linecolor="black",
    square=True
)

plt.title(f"Correlation Heatmap for {outcome_variable.title()}", fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=45, ha="right", rotation_mode='anchor', fontsize=10)
plt.tight_layout()

# ---- Save figure ----

plt.savefig(f"processed_data/42-{outcome_variable}-correlation-heatmap.png", dpi=300)
plt.show()
