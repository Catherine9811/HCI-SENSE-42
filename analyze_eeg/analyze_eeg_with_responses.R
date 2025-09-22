library(lme4)
library(lmerTest)
library(dplyr)
library(readr)
library(ggplot2)

# ---- Setup ----
input_file <- "data/channelwise_absolute_power_with_sleepiness.csv"
output_file <- "data/channelwise_absolute_power_with_sleepiness_results.csv"

# Read the EEG power data
data <- read_csv(input_file)

# Filter participants (same list you used)
data <- filter(data, participant_id %in% c(2, 6, 8, 19, 25, 26, 29, 30, 31, 33, 34, 39, 40, 42))
# data <- filter(data, participant_id %in% c(2, 6, 8))

# Ensure columns are properly typed
data$participant_id <- as.factor(data$participant_id)
data$band <- as.factor(data$band)
data$channel <- as.factor(data$channel)
data$response <- as.numeric(data$response)
data$time <- as.numeric(data$time)
data$power <- as.numeric(data$power)

# Scale time and transform power (same transform you had)
data$time <- data$time / 60
data$power <- 10 * log10(data$power) + 120

# Preset channel lists for each band
band_channels <- list(
  alpha = c("Fp1","Fp2","AF3","AF4","Fz","F3","F4","F7","F8","FC1","FC2","FC5","FC6","P7","P8","Oz","O1","O2"),
  beta  = c("Fp1","Fp2","Fz","Cz","CP1","CP2","Pz","P3","P4","PO3","PO4","Oz"),
  theta = c("Fp1","Fp2","AF3","AF4","Fz","F3","F4","FC1","FC2","FC5","FC6","Cz","C3","C4","CP5","CP6","P7","P8","Oz","O1","O2"),
  delta = c("Fp1","Fp2","AF3","AF4","Fz","F3","F4","FC1","FC2","FC5","FC6","Cz","CP5","CP6","P7","P8","Oz","O1","O2")
)

# Initialize results container
results <- data.frame(
  Band = character(),
  N_obs = integer(),
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

# ---- Loop over bands ----
for (band_name in unique(as.character(data$band))) {
  # skip bands we don't have a preset for
  if (!(band_name %in% names(band_channels))) {
    message("Skipping band (no channel list defined): ", band_name)
    next
  }
  
  # Filter rows for this band and for channels in the preset list
  sel_channels <- band_channels[[band_name]]
  band_data <- filter(data, as.character(band) == band_name, as.character(channel) %in% sel_channels)
  
  # If nothing left, skip
  if (nrow(band_data) == 0) {
    message("No data for band ", band_name, " after channel filtering. Skipping.")
    next
  }
  
  # Group by participant_id, response, time and compute mean power across the selected channels
  # (this averages across channels for the band)
  summarized <- band_data %>%
    group_by(participant_id, response, time) %>%
    # remove outliers within each group
    mutate(
      Q1 = quantile(power, 0.25, na.rm = TRUE),
      Q3 = quantile(power, 0.75, na.rm = TRUE),
      IQR = Q3 - Q1,
      lower = Q1 - 1.5 * IQR,
      upper = Q3 + 1.5 * IQR
    ) %>%
    filter(power >= lower, power <= upper) %>%
    summarize(
      mean_power = mean(power, na.rm = TRUE),
      n_channels = n(),
      .groups = "drop"
    )
  
  # Optionally keep only groups with at least one channel contributing (n_channels > 0)
  summarized <- filter(summarized, n_channels > 0)
  
  # Skip if too few observations overall to fit a mixed model
  if (nrow(summarized) < 10) {
    message("Too few summarized observations for band ", band_name, " (n = ", nrow(summarized), "). Skipping.")
    next
  }
  
  # Fit mixed model: response ~ mean_power + (1 | participant_id)
  model <- tryCatch({
    lmer(response ~ mean_power + (1 | participant_id), data = summarized, REML = FALSE)
  }, error = function(e) {
    message("Model failed for band ", band_name, ": ", e$message)
    NULL
  })
  
  if (is.null(model)) next
  
  # Model summary & statistics
  summary_model <- summary(model)
  conf_int <- tryCatch({
    confint(model, level = 0.95, method = "Wald")
  }, error = function(e) {
    # fallback to profile if Wald fails (or return NA)
    tryCatch(confint(model, level = 0.95), error = function(e2) NA)
  })
  
  # Fit null model for ANOVA comparison
  null_model <- tryCatch({
    lmer(response ~ (1 | participant_id), data = summarized, REML = FALSE)
  }, error = function(e) NULL)
  
  anova_p <- NA
  if (!is.null(null_model)) {
    anova_res <- tryCatch(anova(null_model, model), error = function(e) NULL)
    if (!is.null(anova_res)) {
      anova_p <- anova_res[2, "Pr(>Chisq)"]
    }
  }
  
  # Extract coefficient statistics safely
  coef_exists <- "mean_power" %in% rownames(summary_model$coefficients)
  estimate <- if (coef_exists) summary_model$coefficients["mean_power", "Estimate"] else NA
  pval <- if (coef_exists) summary_model$coefficients["mean_power", "Pr(>|t|)"] else NA
  ci_lower <- if (coef_exists) conf_int["mean_power", 1] else NA
  ci_upper <- if (coef_exists) conf_int["mean_power", 2] else NA
  
  # Append to results
  results <- rbind(results, data.frame(
    Band = band_name,
    N_obs = nrow(summarized),
    Estimate = estimate,
    CI_Lower = ci_lower,
    CI_Upper = ci_upper,
    P_value = pval,
    AIC = AIC(model),
    BIC = BIC(model),
    NegLogLik = as.numeric(-logLik(model)),
    ANOVA_P_value = anova_p,
    stringsAsFactors = FALSE
  ))
  
  # If the ANOVA p-value is significant, produce a quick plot (like you had before)
  if (!is.na(anova_p) && anova_p <= 0.05) {
    p <- ggplot(summarized, aes(x = response, y = mean_power, color = participant_id)) +
      geom_point(alpha = 0.6) +
      labs(
        title = paste("Band-level mean power vs. Response -", band_name),
        y = expression("EEG mean power (dB)"),
        x = "Response",
        color = "Participant"
      ) +
      theme_minimal()
    print(p)
  }
}

# ---- Save Results ----
write_csv(results, output_file)
cat("Results saved to:", output_file, "\n")
