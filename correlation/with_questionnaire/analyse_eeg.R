library(glue)
library(lmerTest)
library(lattice)
library(ggplot2)
library(sjPlot)

# outcome_variable <- "delta"
# income_variable <- "attentiveness"


# Initialize an empty list to store results
results <- data.frame(
  Predictor = character(),
  Outcome = character(),
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


for (outcome_variable in c("alpha", "beta", "theta", "delta", "gamma")) {
  for (income_variable in c("sleepiness", "attentiveness", "effort", "temporal_demand", "performance", "mental_demand", "frustration")) {
    
    # Read the data from the CSV file
    data <- read.csv(paste("processed_data/event7to9/42-", income_variable, ".csv", sep=""), sep=",")
    
    # data <- filter(data, participant %in% c(1, 2, 6, 8, 19, 25, 26, 29, 30, 31, 33, 34, 39, 40, 42))
    
    data[[outcome_variable]] <- as.numeric(data[[outcome_variable]])
    
    output_file <- paste("processed_data/event7to9/42-", outcome_variable, "-", income_variable, "-output.csv", sep="")
    
    # Define the predictors
    predictors <- c(
      income_variable,
      "time"
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
        Outcome = outcome_variable,
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
    # 
    # # Save the results to a CSV file
    # write.csv(results, file = output_file, row.names = FALSE)
    # 
    # cat("Results saved to:", output_file, "\n")
  }
}

# Filter significant rows
sig_results <- subset(results, ANOVA_P_value < 0.05)

cat("Significant results (ANOVA_p_value < 0.05)\n")
print(sig_results)


