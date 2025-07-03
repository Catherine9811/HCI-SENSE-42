library(dplyr)
library(tidyr)
library(ggplot2)
library(ez)
library(readr)

# Input/output setup
outcome_variable <- "usage_hours"
data <- read.csv(paste0("processed_data/behavioural/42-", outcome_variable, ".csv"))

output_file <- paste0("processed_data/behavioural/42-", outcome_variable, "-output.csv")

# Get unique variable names
variable_names <- unique(data$name)

# Prepare results container
results <- data.frame(
  Name = character(),
  F_value = numeric(),
  Df_num = numeric(),
  Df_den = numeric(),
  P_value = numeric(),
  stringsAsFactors = FALSE
)

# Loop over variables
for (var_name in variable_names) {
  subset_data <- data %>% filter(name == var_name)
  
  # Ensure participant and hours are factors
  subset_data <- subset_data %>%
    mutate(participant = as.factor(participant),
           hours = as.factor(hours))
  
  # Aggregate: mean per participant × hours
  aggregated <- subset_data %>%
    group_by(participant, hours) %>%
    summarise(value = mean(value), .groups = "drop")
  
  if (nrow(aggregated) > 0) {
    # Run ANOVA
    anova_result <- tryCatch({
      model <- aov(value ~ hours, data = aggregated)
      summary_model <- summary(model)[[1]]
      
      F_val <- summary_model$`F value`[1]
      Df_num <- summary_model$Df[1]
      Df_den <- summary_model$Df[2]
      P_val <- summary_model$`Pr(>F)`[1]
      
      # Summary stats
      summary_data <- aggregated %>%
        group_by(hours) %>%
        summarise(mean = mean(value),
                  se = sd(value) / sqrt(n()),
                  .groups = "drop") %>%
        pivot_wider(
          names_from = hours,
          values_from = c(mean, se),
          names_glue = "{hours}_{.value}"
        )
      
      # Combine results
      combined <- cbind(data.frame(
        Name = var_name,
        F_value = F_val,
        Df_num = Df_num,
        Df_den = Df_den,
        P_value = P_val
      ), summary_data)
      
      results <- bind_rows(results, combined)
    }, error = function(e) {
      message(glue::glue("ANOVA failed for variable: {var_name} - {e$message}"))
    })
  }
}

# Save results
write.csv(results, output_file, row.names = FALSE)
cat("Results saved to:", output_file, "\n")
