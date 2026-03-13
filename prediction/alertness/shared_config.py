from __future__ import annotations

from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed_data" / "42-alertness-multimodal.csv"

TARGET_COL = "sleepiness"
GROUP_COL = "participant"
RANDOM_STATE = 42


# Feature groups, translated from the R category_list definition
FEATURE_GROUPS: Dict[str, List[str]] = {
    "physiological_eeg": [
        "alpha",
        "beta",
        "theta",
        "delta",
        "gamma",
    ],
    "physiological_ecg": [
        "cardiac_rr_interval_mean",
        "cardiac_rr_interval_var",
    ],
    "physiological_resp": [
        "respiratory_inhalation_duration_mean",
        "respiratory_inhalation_duration_var",
        "respiratory_exhalation_duration_mean",
        "respiratory_exhalation_duration_var",
    ],
    "behavioural": [
        "head_pose_variation_mean",
        "head_pose_movement_mean",
        "head_pitch_variation_mean",
        "head_roll_variation_mean",
        "head_yaw_variation_mean",
        "blink_times_mean",
        "look_down_times_mean",
    ],
    "interaction_hci": [
        "mouse_double_click_duration_mean",
        "mouse_double_click_duration_var",
        "mouse_double_click_movement_mean",
        "mouse_double_click_movement_var",
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
        "mouse_open_notification_unintended_clicks_var",
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
        "keyboard_side_by_side_typing_efficiency_var",
    ],
    "participant_traits": [
        "age",
        "gender",
        "ethnic_group",
        "occupation",
        "education_level",
        "which_languages_are_you_comfortable_using_select_all_that_apply",
    ],
    "computer_use": [
        "how_comfortable_are_you_with_using_computers",
        "primary_reason_for_computer_use_select_all_that_apply",
        "computer_usage_on_average_how_many_hours_per_day_do_you_spend_using_a_computer",
        "what_operating_systems_do_you_use_most_frequently_select_all_that_apply",
        "which_forms_of_computer_are_you_most_familiar_with_select_all_that_apply",
        "which_input_devices_are_you_most_familiar_with_select_all_that_apply",
        "how_frequently_do_you_use_a_keyboard_for_computer_related_tasks",
        "which_keyboard_layouts_do_you_use_most_frequently_for_english_select_all_that_apply",
    ],
    "sleep_quality": [
        "sleep_patterns_on_average_how_many_hours_of_sleep_do_you_get_per_night",
        "psqi_51_how_often_have_you_had_trouble_sleeping_because_not_during_the_past_month",
        "do_you_have_any_known_sleep_disorders_or_difficulties_fallingstaying_asleep",
        "ess_total",
        "do_you_consume_any_substances_that_could_affect_alertness_eg_caffeine_alcohol_medications_etc",
        "psqi_1_what_time_have_you_usually_gone_to_bed_at_night",
        "psqi_2_how_long_has_it_usually_taken_you_to_fall_asleep_each_night",
        "psqi_3_what_time_have_you_usually_gotten_up_in_the_morning",
        "psqi_4_how_many_hours_of_actual_sleep_did_you_get_at_night",
        "psqi_54_how_often_have_you_had_trouble_sleeping_because_three_or_more_times_a_week",
        "psqi_6_how_often_have_you_taken_medicine_to_help_you_sleep",
        "psqi_10_do_you_have_a_bed_partner_or_room_mate",
        "ess_1_situation_dozing_while_sitting_and_reading",
        "ess_2_situation_dozing_while_watching_tv",
        "ess_3_situation_dozing_while_sitting_inactive_in_a_public_place",
        "ess_4_situation_dozing_while_as_a_passenger_in_a_car_for_an_hour_without_a_break",
        "ess_5_situation_dozing_while_lying_down_to_rest_in_the_afternoon_when_circumstances_permit",
        "ess_6_situation_dozing_while_sitting_and_talking_to_someone",
        "ess_7_situation_dozing_while_sitting_quitely_after_a_lunch_without_alcohol",
        "ess_8_situation_dozing_while_in_a_car_while_stopped_for_a_few_minutes_in_the_traffic",
        "psqi_7_how_often_have_you_had_trouble_staying_awake_while_driving_eating_meals_or_engaging_in_social_activity",
        "psqi_8_how_much_of_a_problem_has_it_been_for_you_to_keep_up_enough_enthusiasm_to_get_things_done",
        "psqi_9_how_would_you_rate_your_sleep_quality_overall",
        "psqi_52_how_often_have_you_had_trouble_sleeping_because_less_than_once_a_week",
        "psqi_53_how_often_have_you_had_trouble_sleeping_because_once_or_twice_a_week",
    ],
    "engagement": [
        # "time",
        "testing_time",
        "testing_order",
        "outside_temperature",
        "do_you_wear_corrective_lenses_glassescontact_lenses_or_have_any_vision_impairments",
        "handedness",
        "have_you_ever_been_diagnosed_with_any_neurological_or_psychiatric_conditions",
    ],
}

# Combined group: interaction_hci + participant traits + computer_use + sleep quality + engagement
COMBINED_KEYS = [
    "interaction_hci",
    "participant_traits",
    "computer_use",
    "sleep_quality",
    "engagement",
]
combined_features = sorted(
    {
        f
        for k in COMBINED_KEYS
        for f in FEATURE_GROUPS.get(k, [])
    }
)
FEATURE_GROUPS["mouse_keyboard_traits_sleep_engagement"] = combined_features

# Add a combined "all_features" group for overall performance
all_feature_names = sorted({f for group in FEATURE_GROUPS.values() for f in group})
FEATURE_GROUPS["all_features"] = all_feature_names
