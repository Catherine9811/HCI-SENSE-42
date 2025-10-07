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
  
  # Group by participant_id, response, time and compute mean power across channels
  summarized <- band_data %>%
    group_by(participant_id, response, time) %>%
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
  
  summarized <- filter(summarized, n_channels > 0)
  
  if (nrow(summarized) < 10) {
    message("Too few summarized observations for band ", band_name, " (n = ", nrow(summarized), "). Skipping.")
    next
  }
  
  # ---- Plot band power vs. time ----
  p_time <- ggplot(summarized, aes(x = time, y = mean_power, group = participant_id, color = participant_id)) +
    geom_line(alpha = 1.0, linewidth = 2.0) +   # each participant trajectory
    # stat_summary(aes(group = 1), fun = mean, geom = "line", color = "black", size = 1.2) + # group mean
    labs(
      title = paste("Band power over time -", band_name),
      x = "Time (minutes)",
      y = "Mean power (dB)",
      color = "Participant"
    ) +
    theme_minimal()
  print(p_time)
}
