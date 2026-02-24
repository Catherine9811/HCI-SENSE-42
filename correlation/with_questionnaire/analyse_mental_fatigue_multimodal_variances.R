#############################################################
# ---- FULL MODEL + partR2 VARIANCE DECOMPOSITION ----------
#############################################################

library(partR2)
library(glue)
library(lmerTest)
library(lattice)
library(ggplot2)
library(sjPlot)
library(dplyr)
library(future)

outcome_variables <- c("performance", "temporal_demand", "effort", "frustration", "mental_demand")
outcome_variable <- "tlx"

# Read the data
data <- read.csv(paste("processed_data/42-sleepiness-multimodal.csv", sep=""), sep=",")
# Ensure selected variables are numeric
data[outcome_variables] <- lapply(data[outcome_variables], as.numeric)

# Create tlx as row-wise sum of specified variables
data[[outcome_variable]] <- rowSums(data[outcome_variables], na.rm = TRUE)

output_file <- paste("processed_data/42-", outcome_variable, "-multimodal-output.csv", sep="")

# Predictor selection
exclude_cols <- c("participant", "initiation", "attentiveness", "sleepiness", "tlx", "performance", "temporal_demand", "effort", "frustration", "mental_demand",
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
                  "testing_time", "testing_order")


predictors <- setdiff(names(data), exclude_cols)

# data[predictors] <- lapply(data[predictors], function(x) {
#   if (is.numeric(x)) {
#     as.numeric(scale(x))
#   } else {
#     x
#   }
# })

# ---- 1. Fit initial model with all predictors ----
full_formula <- as.formula(
  paste(outcome_variable, "~", paste(predictors, collapse = " + "), " ")
)

full_model <- lm(full_formula, data = data)

# mf <- model.frame(full_formula, data=data)
# 
# X_full <- model.matrix(full_formula, mf)
# 
# qrX <- qr(X_full)
# 
# bad_idx <- qrX$pivot[(qrX$rank + 1):ncol(X_full)]
# 
# colnames(X_full)[bad_idx]

summary(full_model)
tab_model(full_model)
print(full_model, correlation = TRUE)

# ---- 2. Identify significant predictors ----
coef_table <- summary(full_model)$coefficients

# Extract p-values for fixed effects except intercept
fixed_effects <- coef_table[-1, , drop = FALSE]  # drop intercept
significant_predictors <- rownames(fixed_effects)[fixed_effects[, "Pr(>|t|)"] < 0.05]

cat("\nSignificant predictors:\n")
print(significant_predictors)

if (length(significant_predictors) == 0) {
  stop("No significant predictors found. Cannot run partR2.")
}

category_list <- list(
  physiological_eeg = c("alpha", "beta", "theta", "delta", "gamma"),
  physiological_ecg = c("cardiac_rr_interval_mean", "cardiac_rr_interval_var"),
  physiological_resp = c("respiratory_inhalation_duration_mean", "respiratory_inhalation_duration_var", "respiratory_exhalation_duration_mean", "respiratory_exhalation_duration_var"),
  behavioural = c("head_pose_variation_mean", "head_pose_movement_mean", "head_pitch_variation_mean", "head_roll_variation_mean", "head_yaw_variation_mean", "blink_times_mean", "look_down_times_mean"),
  interaction_mouse = c(
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
  ),
  interaction_keyboard = c(
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
  ),
  participant_traits = c(
    "age",
    "gender",
    "ethnic_group",
    "occupation",
    "education_level",
    "which_languages_are_you_comfortable_using_select_all_that_apply"
  ),
  computer_use = c(
    "how_comfortable_are_you_with_using_computers",
    "primary_reason_for_computer_use_select_all_that_apply",
    "what_operating_systems_do_you_use_most_frequently_select_all_that_apply",
    "which_forms_of_computer_are_you_most_familiar_with_select_all_that_apply",
    "which_input_devices_are_you_most_familiar_with_select_all_that_apply",
    "which_keyboard_layouts_do_you_use_most_frequently_for_english_select_all_that_apply"
  ),
  sleep_quality = c(
    "sleep_patterns_on_average_how_many_hours_of_sleep_do_you_get_per_night",
    "psqi_51_how_often_have_you_had_trouble_sleeping_because_not_during_the_past_month",
    "do_you_have_any_known_sleep_disorders_or_difficulties_fallingstaying_asleep",
    "ess_total",
    "do_you_consume_any_substances_that_could_affect_alertness_eg_caffeine_alcohol_medications_etc",
    "psqi_7_how_often_have_you_had_trouble_staying_awake_while_driving_eating_meals_or_engaging_in_social_activity",
    "psqi_8_how_much_of_a_problem_has_it_been_for_you_to_keep_up_enough_enthusiasm_to_get_things_done",
    "psqi_9_how_would_you_rate_your_sleep_quality_overall",
    "psqi_52_how_often_have_you_had_trouble_sleeping_because_less_than_once_a_week",
    "psqi_53_how_often_have_you_had_trouble_sleeping_because_once_or_twice_a_week"
  ),
  engagement = c(
    "time",
    "outside_temperature",
    "do_you_wear_corrective_lenses_glassescontact_lenses_or_have_any_vision_impairments",
    "handedness",
    "how_comfortable_are_you_with_using_computers",
    "primary_reason_for_computer_use_select_all_that_apply",
    "computer_usage_on_average_how_many_hours_per_day_do_you_spend_using_a_computer",
    "which_languages_are_you_comfortable_using_select_all_that_apply",
    "what_operating_systems_do_you_use_most_frequently_select_all_that_apply",
    "which_forms_of_computer_are_you_most_familiar_with_select_all_that_apply",
    "which_input_devices_are_you_most_familiar_with_select_all_that_apply",
    "how_frequently_do_you_use_a_keyboard_for_computer_related_tasks",
    "which_keyboard_layouts_do_you_use_most_frequently_for_english_select_all_that_apply"
  )
)

library(dplyr)
library(stringr)

# helper function
compute_r2 <- function(preds) {
  if (length(preds) == 0) {
    return(0)
  }
  f <- as.formula(
    paste(outcome_variable, "~", paste(preds, collapse = " + "))
  )
  m <- lm(f, data = data)
  summary(m)$r.squared
}

category_names <- names(category_list)

all_predictors <- unique(unlist(category_list))

total_r2 <- compute_r2(all_predictors)

# unique contributions of single categories
single_rows <- lapply(category_names, function(cat) {
  cat_vars          <- category_list[[cat]]
  preds_without_cat <- setdiff(all_predictors, cat_vars)
  r2_without_cat    <- compute_r2(preds_without_cat)
  unique_cat        <- total_r2 - r2_without_cat
  data.frame(
    term     = cat,
    estimate = unique_cat,
    stringsAsFactors = FALSE
  )
})
single_rows <- do.call(rbind, single_rows)

# unique contributions of pairwise combinations of categories
if (length(category_names) > 1) {
  pair_list <- list()
  idx <- 1
  for (i in 1:(length(category_names) - 1)) {
    for (j in (i + 1):length(category_names)) {
      cat1 <- category_names[i]
      cat2 <- category_names[j]
      vars1 <- category_list[[cat1]]
      vars2 <- category_list[[cat2]]
      preds_without_both <- setdiff(all_predictors, c(vars1, vars2))
      r2_without_both    <- compute_r2(preds_without_both)
      unique_ab          <- total_r2 - r2_without_both
      pair_list[[idx]] <- data.frame(
        term     = paste(cat1, "+", cat2),
        estimate = unique_ab,
        stringsAsFactors = FALSE
      )
      idx <- idx + 1
    }
  }
  pair_rows <- do.call(rbind, pair_list)
} else {
  pair_rows <- data.frame(term = character(0), estimate = numeric(0))
}

r2_table <- rbind(
  data.frame(term = "Full", estimate = total_r2, stringsAsFactors = FALSE),
  single_rows,
  pair_rows
)

write.csv(
  r2_table,
  paste0("processed_data/42-", outcome_variable, "-multimodal-partR2-unique.csv"),
  row.names = FALSE
)

cat("Category-level R² table saved.\n")


unique_r2 <- r2_table %>% 
  filter(term != "Full") %>%
  filter(!str_detect(term, "\\+")) %>% 
  select(term, estimate)

# ---- PAIRWISE BLOCK CONTRIBUTIONS (terms like "A + B") ----
pair_blocks <- r2_table %>% 
  filter(str_count(term, "\\+") == 1) %>%
  mutate(vars = str_split(term, "\\+"))

# Compute pairwise overlaps
pair_overlaps <- pair_blocks %>% 
  rowwise() %>% 
  mutate(
    var1 = str_trim(vars[[1]]),
    var2 = str_trim(vars[[2]]),
    overlap = estimate - (
      unique_r2$estimate[unique_r2$term == var1] +
        unique_r2$estimate[unique_r2$term == var2]
    )
  ) %>%
  ungroup() %>%
  select(var1, var2, overlap)

# ---- FULL R² AND GLOBAL SHARED VARIANCE ----
total_r2 <- r2_table %>% filter(term == "Full") %>% pull(estimate)

global_shared <- total_r2 - sum(unique_r2$estimate)

global_shared_df <- data.frame(
  term = "Shared (total)",
  estimate = global_shared
)

# plot_unique <- ggplot(unique_r2, aes(x = term, y = estimate, fill = term)) +
#   geom_col() +
#   labs(
#     title = "Unique Variance Explained",
#     x = "Predictor",
#     y = "Unique R²"
#   ) +
#   theme_minimal() +
#   theme(legend.position = "none")
# 
# plot_unique
# 
# plot_pairwise <- ggplot(pair_overlaps, 
#                         aes(x = interaction(var1, var2), y = overlap, fill = interaction(var1, var2))) +
#   geom_col() +
#   labs(
#     title = "Pairwise Shared Variance (Overlap)",
#     x = "Predictor Pair",
#     y = "Overlap R²"
#   ) +
#   theme_minimal() +
#   theme(legend.position = "none")
# 
# plot_pairwise


library(ggplot2)
library(dplyr)

# Compute shared & unexplained
unique_sum <- sum(unique_r2$estimate)
shared_r2 <- total_r2 - unique_sum
unexplained_r2 <- 1 - total_r2   # full pie covers 100%

pie_df <- bind_rows(
  unique_r2 %>% rename(value = estimate),
  data.frame(term = "Shared variance", value = shared_r2),
  data.frame(term = "Unexplained variance", value = unexplained_r2)
)

# Colors:
# - Unique predictors: automatic ggplot palette
# - Shared = dark gray
# - Unexplained = light gray
colors <- c(
  # Physiological (cold colours)
  "Physiological (EEG)"         = "#1F78B4",  # cold blue
  "Physiological (ECG)"         = "#33A1C9",  # cold cyan
  "Physiological (Respiration)" = "#5CC8FF",  # light icy blue
  
  # Behavioural (neutral colours)
  "Behavioural (Webcam)"        = "#999999",  # medium neutral gray
  
  # Interaction (warm colours)
  "Mouse Interaction"           = "#E69F00",  # warm orange
  "Keyboard Interaction"        = "#D55E00",  # warm red-orange
  
  "Individual Traits"           = "#98df8a",
  "Computer Use Proficiency"    = "#2ca02c",
  "Quality of Sleep"            = "#006400",
  
  # Engagement & context (distinct colour)
  "Engagement & Context"        = "#9467BD",
  
  # Shared / Unexplained
  "Shared Variance"             = "gray40",
  "Unexplained Variance"        = "gray85"
)

rename_vars <- c(
  "physiological_eeg"    = "Physiological (EEG)",
  "physiological_ecg"    = "Physiological (ECG)",
  "physiological_resp"   = "Physiological (Respiration)",
  "behavioural"          = "Behavioural (Webcam)",
  "interaction_mouse"    = "Mouse Interaction",
  "interaction_keyboard" = "Keyboard Interaction",
  "participant_traits"   = "Individual Traits",
  "computer_use"         = "Computer Use Proficiency",
  "sleep_quality"        = "Quality of Sleep",
  "engagement"           = "Engagement & Context",
  "Shared variance"      = "Shared Variance",
  "Unexplained variance" = "Unexplained Variance"
)

pie_df$term <- rename_vars[pie_df$term]

# pie_df <- pie_df %>%
#   mutate(
#     pct = value / sum(value),                     # percentage
#     pct_label = scales::percent(pct, accuracy = 1),
#     cum = cumsum(pct),
#     pos = cum - pct / 2                           # midpoint for labels
#   )

plot_pie <- ggplot(pie_df, aes(x = "", y = value, fill = term)) +
  geom_col(color = "white", width = 1) +
  # geom_text(aes(label = pct_label, y = pos), color = "black", size = 4) +
  coord_polar(theta = "y") +
  scale_fill_manual(values = colors, na.value = "steelblue") +
  labs(
    title = glue("Variance explained in {toupper(outcome_variable)}"),
    fill  = "Predictors"
  ) +
  theme_void() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16)
  )

plot_pie

write.csv(pie_df,
          paste0("processed_data/42-", outcome_variable, "-multimodal-individual-contributions.csv"),
          row.names = FALSE)


library(tidyr)

vars <- unique(unique_r2$term)

# Diagonal values (unique contributions)
diag_df <- unique_r2 %>%
  rename(value = estimate) %>%
  mutate(var1 = term, var2 = term) %>%
  select(var1, var2, value)

# Off-diagonal values (pairwise overlap)
off_df <- pair_overlaps %>% rename(value = overlap)

# Combine into one matrix-like table
matrix_df <- bind_rows(diag_df, off_df)

plot_matrix <- ggplot(matrix_df, aes(x = var1, y = var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "red", mid = "white", high = "blue",
    midpoint = 0,
    name = "R²"
  ) +
  geom_text(aes(label = round(value, 3)), size = 4) +
  labs(
    title = "Unique and Shared Variance (Correlation-Style)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title = element_blank(),
    plot.title = element_text(size = 16, hjust = 0.5)
  )

plot_matrix

write.csv(matrix_df,
          paste0("processed_data/42-", outcome_variable, "-multimodal-shared-contributions.csv"),
          row.names = FALSE)
