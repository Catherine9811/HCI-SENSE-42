library(glue)
library(lmerTest)
library(lattice)
library(ggplot2)
library(sjPlot)

outcome_variable <- "sleepiness"

# Read the data from the CSV file
data <- read.csv(paste("processed_data/behavioural/42-", outcome_variable, ".csv", sep=""), sep=",")

data[[outcome_variable]] <- as.numeric(data[[outcome_variable]])

output_file <- paste("processed_data/behavioural/42-", outcome_variable, "-output.csv", sep="")

# Define the predictors
predictors <- c(
  "mouse_double_click_distance_mean", "mouse_double_click_distance_var", 
  "mouse_drag_distance_mean", "mouse_drag_distance_var", 
  "mouse_drop_distance_mean", "mouse_drop_distance_var", 
  "mouse_taskbar_navigation_efficiency_mean", "mouse_taskbar_navigation_efficiency_var", 
  "mouse_toolbar_navigation_efficiency_mean", "mouse_toolbar_navigation_efficiency_var", 
  "mouse_selection_coverage_mean", "mouse_selection_coverage_var", 
  "mouse_folder_navigation_speed_mean", "mouse_folder_navigation_speed_var", 
  "mouse_toolbar_navigation_speed_mean", "mouse_toolbar_navigation_speed_var", 
  "mouse_confirm_dialog_duration_mean", "mouse_confirm_dialog_duration_var", 
  "mouse_notification_duration_mean", "mouse_notification_duration_var", 
  "mouse_open_folder_duration_mean", "mouse_open_folder_duration_var", 
  "mouse_drag_folder_duration_mean", "mouse_drag_folder_duration_var", 
  "mouse_close_window_duration_mean", "mouse_close_window_duration_var", 
  "mouse_grouped_selection_duration_mean", "mouse_grouped_selection_duration_var", 
  "keyboard_shadow_typing_duration_mean", "keyboard_shadow_typing_duration_var", 
  "keyboard_side_by_side_typing_duration_mean", "keyboard_side_by_side_typing_duration_var", 
  "keyboard_shadow_typing_error_mean", "keyboard_shadow_typing_error_var", 
  "keyboard_side_by_side_typing_error_mean", "keyboard_side_by_side_typing_error_var", 
  "keyboard_typing_speed_mean", "keyboard_typing_speed_var", 
  "keyboard_space_key_pressed_duration_mean", "keyboard_space_key_pressed_duration_var", 
  "keyboard_space_key_typing_duration_mean", "keyboard_space_key_typing_duration_var", 
  "keyboard_pressed_duration_mean", "keyboard_pressed_duration_var", 
  "keyboard_shadow_typing_efficiency_mean", "keyboard_shadow_typing_efficiency_var",
  "keyboard_side_by_side_typing_efficiency_mean", "keyboard_side_by_side_typing_efficiency_var",
  "mouse_open_file_manager_duration_mean", "mouse_open_file_manager_duration_var",
  "mouse_open_trash_bin_duration_mean", "mouse_open_trash_bin_duration_var",
  "mouse_open_notes_duration_mean", "mouse_open_notes_duration_var",
  "mouse_open_browser_duration_mean", "mouse_open_browser_duration_var",
  "mouse_open_folder_clicking_duration_mean", "mouse_open_folder_clicking_duration_var",
  "mouse_close_window_clicking_duration_mean", "mouse_close_window_clicking_duration_var",
  "mouse_open_folder_unintended_clicks_mean", "mouse_open_folder_unintended_clicks_var",
  "mouse_close_window_unintended_clicks_mean", "mouse_close_window_unintended_clicks_var",
  "time"
)

# Initialize an empty list to store results
results <- data.frame(
  Predictor = character(),
  Estimate = numeric(),
  CI_Lower = numeric(),
  CI_Upper = numeric(),
  P_value = numeric(),
  AIC = numeric(),
  BIC = numeric(),
  NegLogLik = numeric(),
  ANOVA_P_value = numeric(),
  stringsAsFactors = FALSE
)

# Loop through each predictor
for (predictor in predictors) {
  # Fit the multilevel model
  formula <- as.formula(paste(outcome_variable, "~", predictor, "+ (1 | participant)"))
  model <- lmer(formula, data = data, REML = FALSE)
  
  # Get model summary
  model_summary <- summary(model)
  
  # Extract statistics for the predictor
  estimate <- model_summary$coefficients[predictor, "Estimate"]
  conf_int <- confint(model, level = 0.95)[predictor, ]
  p_value <- model_summary$coefficients[predictor, "Pr(>|t|)"]
  
  # Calculate AIC, BIC, and negative log-likelihood
  aic <- AIC(model)
  bic <- BIC(model)
  neg_log_lik <- -logLik(model)
  
  # Fit null model for ANOVA
  null_model <- lmer(paste(outcome_variable, "~ (1 | participant)"), data = data, REML = FALSE)
  anova_result <- anova(null_model, model)
  anova_p_value <- anova_result[2, "Pr(>Chisq)"]
  
  # Store the results
  results <- rbind(results, data.frame(
    Predictor = predictor,
    Estimate = estimate,
    CI_Lower = conf_int[1],
    CI_Upper = conf_int[2],
    P_value = p_value,
    AIC = aic,
    BIC = bic,
    NegLogLik = as.numeric(neg_log_lik),
    ANOVA_P_value = anova_p_value,
    stringsAsFactors = FALSE
  ))
}

# Save the results to a CSV file
write.csv(results, file = output_file, row.names = FALSE)

cat("Results saved to:", output_file, "\n")
