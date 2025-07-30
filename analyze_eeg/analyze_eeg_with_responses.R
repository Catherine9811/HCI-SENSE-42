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
data <- filter(data, participant_id %in% c(2, 6, 8, 19, 25, 26, 29, 30, 31, 33, 34, 39, 40, 42))
# Ensure columns are properly typed
data$participant_id <- as.factor(data$participant_id)
data$band <- as.factor(data$band)
data$channel <- as.factor(data$channel)
data$response <- as.numeric(data$response)
data$time <- as.numeric(data$time)
data$power <- as.numeric(data$power)
data$time <- data$time / 7200
data$power <- 10 * log(data$power) + 120
# Initialize results container
results <- data.frame(
  Band = character(),
  Channel = character(),
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

# ---- Loop over band × channel ----
for (band_name in unique(data$band)) {
  for (channel_name in unique(data$channel)) {
    subset_data <- filter(data, band == band_name, channel == channel_name)
    
    # Skip if too few observations
    if (nrow(subset_data) < 10) next
    
    # Fit mixed model: response ~ power + (1 | participant_id)
    model <- tryCatch({
      lmer(response ~ power + (1 | participant_id), data = subset_data, REML = FALSE)
    }, error = function(e) NULL)
    
    if (!is.null(model)) {
      summary_model <- summary(model)
      conf_int <- confint(model, level = 0.95, method = "Wald")
      null_model <- lmer(response ~ (1 | participant_id), data = subset_data, REML = FALSE)
      anova_p <- anova(null_model, model)[2, "Pr(>Chisq)"]
      
      if (anova_p <= 0.05) {
        p <- ggplot(subset_data, aes(x = response, y = power, color = participant_id)) +
          geom_point(alpha = 0.6) +
          geom_smooth(method = "lm", se = TRUE, color = "black", linetype = "dashed") +
          labs(
            title = paste("Power vs. Response -", band_name, "-", channel_name),
            y = expression("EEG Power ("*10^-12~W*")"),
            x = "Response",
            color = "Participant"
          ) +
          theme_minimal()
        
        print(p)  # Show plot instead of saving
      }
      
      results <- rbind(results, data.frame(
        Band = band_name,
        Channel = channel_name,
        Estimate = summary_model$coefficients["power", "Estimate"],
        CI_Lower = conf_int["power", 1],
        CI_Upper = conf_int["power", 2],
        P_value = summary_model$coefficients["power", "Pr(>|t|)"],
        AIC = AIC(model),
        BIC = BIC(model),
        NegLogLik = as.numeric(-logLik(model)),
        ANOVA_P_value = anova_p,
        stringsAsFactors = FALSE
      ))
    }
  }
}

# ---- Save Results ----
write_csv(results, output_file)
cat("Results saved to:", output_file, "\n")
