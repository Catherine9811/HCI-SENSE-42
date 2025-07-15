library(glue)
library(dplyr)
library(broom)
library(performance)

outcome_variable <- "attentiveness"
data <- read.csv(paste0("processed_data/behavioural/42-", outcome_variable, ".csv"))

data[[outcome_variable]] <- as.numeric(data[[outcome_variable]])

output_file <- paste0("processed_data/behavioural/42-", outcome_variable, "-fixed-effects-output.csv")

predictors <- c(
  "keyboard_pressed_duration_mean", "mouse_drop_distance_mean",
  "mouse_drag_distance_mean", "mouse_double_click_distance_mean",
  "mouse_drop_distance_var", "mouse_drag_distance_var",
  "mouse_grouped_selection_duration_var", "keyboard_pressed_duration_var",
  "mouse_folder_navigation_speed_mean", "mouse_double_click_distance_var",
  "mouse_selection_coverage_var"
)

correction_methods <- c("none", "mean", "median", "min", "max", "minmax")

results <- data.frame(
  Predictor = character(),
  Correction = character(),
  Estimate = numeric(),
  CI_Lower = numeric(),
  CI_Upper = numeric(),
  P_value = numeric(),
  R2 = numeric(),
  Adj_R2 = numeric(),
  AIC = numeric(),
  BIC = numeric(),
  stringsAsFactors = FALSE
)

for (predictor in predictors) {
  # Compute summaries per participant
  summaries <- data %>%
    group_by(participant) %>%
    summarise(
      mean = mean(.data[[predictor]], na.rm = TRUE),
      median = median(.data[[predictor]], na.rm = TRUE),
      min = min(.data[[predictor]], na.rm = TRUE),
      max = max(.data[[predictor]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(minmax = max - min)
  
  data_with_summaries <- data[, c(predictor, outcome_variable, "participant")] %>%
    left_join(summaries, by = "participant", suffix = c("", "_summary"))
  
  for (correction in correction_methods) {
    # Prepare formula
    if (correction == "none") {
      
    } else if (correction == "minmax") {
      data_with_summaries[predictor] <- (data_with_summaries[predictor] - data_with_summaries["min"]) / data_with_summaries["minmax"]
    }
    else {
      data_with_summaries[predictor] <- data_with_summaries[predictor] - data_with_summaries[correction]
    }
    formula <- as.formula(paste0(outcome_variable, " ~ ", predictor))
    model <- lm(formula, data = data_with_summaries)
    coef_summary <- tidy(model, conf.int = TRUE)
    
    # Extract row for the predictor
    row_name <- if (correction == "none") predictor else predictor
    coef_row <- coef_summary %>% filter(term == row_name)
    
    if (nrow(coef_row) == 0) next  # skip if predictor is not in model
    
    # Model performance
    r2 <- performance::r2(model)$R2
    adj_r2 <- summary(model)$adj.r.squared
    aic <- AIC(model)
    bic <- BIC(model)
    
    results <- rbind(results, data.frame(
      Predictor = predictor,
      Correction = correction,
      Estimate = coef_row$estimate,
      CI_Lower = coef_row$conf.low,
      CI_Upper = coef_row$conf.high,
      P_value = coef_row$p.value,
      R2 = r2,
      Adj_R2 = adj_r2,
      AIC = aic,
      BIC = bic,
      stringsAsFactors = FALSE
    ))
  }
}

write.csv(results, file = output_file, row.names = FALSE)
cat("Fixed effects model results saved to:", output_file, "\n")

# library(glue)
# library(lmerTest)
# library(lattice)
# library(ggplot2)
# library(sjPlot)
# library(dplyr)
# library(performance)  # For r2() and icc()
# 
# outcome_variable <- "sleepiness"
# data <- read.csv(paste0("processed_data/behavioural/42-", outcome_variable, ".csv"))
# output_file <- paste0("processed_data/behavioural/42-", outcome_variable, "-output-between-subject.csv")
# 
# predictors <- c(
#   "keyboard_pressed_duration_mean", "mouse_drop_distance_mean",
#   "mouse_drag_distance_mean", "mouse_double_click_distance_mean",
#   "mouse_drop_distance_var", "mouse_drag_distance_var",
#   "mouse_grouped_selection_duration_var", "keyboard_pressed_duration_var",
#   "mouse_folder_navigation_speed_mean", "mouse_double_click_distance_var",
#   "mouse_selection_coverage_var"
# )
# 
# # Correction types to add
# correction_methods <- c("none", "mean", "median", "min", "max", "minmax")
# 
# # Prepare result storage
# results <- data.frame(
#   Predictor = character(),
#   Correction = character(),
#   Estimate = numeric(),
#   CI_Lower = numeric(),
#   CI_Upper = numeric(),
#   P_value = numeric(),
#   AIC = numeric(),
#   BIC = numeric(),
#   NegLogLik = numeric(),
#   ANOVA_P_value = numeric(),
#   R2_marginal = numeric(),
#   R2_conditional = numeric(),
#   ICC = numeric(),
#   stringsAsFactors = FALSE
# )
# 
# for (predictor in predictors) {
#   
#   # Compute subject-level summaries for each correction method
#   summaries <- data %>%
#     group_by(participant) %>%
#     summarise(
#       mean = mean(.data[[predictor]], na.rm = TRUE),
#       median = median(.data[[predictor]], na.rm = TRUE),
#       min = min(.data[[predictor]], na.rm = TRUE),
#       max = max(.data[[predictor]], na.rm = TRUE)
#     ) %>%
#     mutate(minmax = min + max)
#   
#   # Join to main data
#   data_with_summaries <- data %>%
#     left_join(summaries, by = "participant", suffix = c("", "_summary"))
#   
#   for (correction in correction_methods) {
#     if (correction == "none") {
#       formula <- as.formula(paste(outcome_variable, "~", predictor, "+ (1 | participant)"))
#     } else {
#       formula <- as.formula(paste0(
#         outcome_variable, " ~ ", predictor, " + ", correction, " + (1 | participant)"
#       ))
#     }
#     
#     # Fit full model
#     model <- lmer(formula, data = data_with_summaries, REML = FALSE)
#     model_summary <- summary(model)
#     
#     # Handle variable name for coefficient extraction
#     target_var <- if (correction == "none") predictor else predictor
#     
#     # Extract stats
#     estimate <- model_summary$coefficients[target_var, "Estimate"]
#     conf_int <- confint(model, level = 0.95)[target_var, ]
#     p_value <- model_summary$coefficients[target_var, "Pr(>|t|)"]
#     aic <- AIC(model)
#     bic <- BIC(model)
#     neg_log_lik <- -logLik(model)
#     
#     # Null model
#     null_model <- lmer(paste(outcome_variable, "~ (1 | participant)"), data = data_with_summaries, REML = FALSE)
#     anova_result <- anova(null_model, model)
#     anova_p_value <- anova_result[2, "Pr(>Chisq)"]
#     
#     # R2 and ICC
#     r2_vals <- suppressWarnings(performance::r2(model))
#     icc_val <- suppressWarnings(performance::icc(model)$ICC_adjusted)
#     
#     # Save result
#     results <- rbind(results, data.frame(
#       Predictor = predictor,
#       Correction = correction,
#       Estimate = estimate,
#       CI_Lower = conf_int[1],
#       CI_Upper = conf_int[2],
#       P_value = p_value,
#       AIC = aic,
#       BIC = bic,
#       NegLogLik = as.numeric(neg_log_lik),
#       ANOVA_P_value = anova_p_value,
#       R2_marginal = r2_vals$R2_marginal,
#       R2_conditional = r2_vals$R2_conditional,
#       ICC = icc_val,
#       stringsAsFactors = FALSE
#     ))
#   }
# }
# 
# write.csv(results, file = output_file, row.names = FALSE)
# cat("Results saved to:", output_file, "\n")
