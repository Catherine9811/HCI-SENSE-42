library(glue)
library(lmerTest)
library(lattice)
library(ggplot2)
library(sjPlot)

outcome_variable <- "sleepiness"

# Read the data from the CSV file
data <- read.csv(paste("processed_data/cardiac/9-", outcome_variable, ".csv", sep=""), sep=",")

# Define the predictors
predictors <- c(
  "cardiac_rr_interval_mean", 
  "cardiac_rr_interval_var"
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
  formula <- as.formula(paste(predictor, "~", "time + (1 | participant)"))
  model <- lmer(formula, data = data, REML = FALSE)
  
  # Get model summary
  model_summary <- summary(model)
  
  # Extract statistics for the predictor
  estimate <- model_summary$coefficients["time", "Estimate"]
  conf_int <- confint(model, level = 0.95)["time", ]
  p_value <- model_summary$coefficients["time", "Pr(>|t|)"]
  
  # Calculate AIC, BIC, and negative log-likelihood
  aic <- AIC(model)
  bic <- BIC(model)
  neg_log_lik <- -logLik(model)
  
  # Fit null model for ANOVA
  null_model <- lmer(paste(predictor, "~ (1 | participant)"), data = data, REML = FALSE)
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

