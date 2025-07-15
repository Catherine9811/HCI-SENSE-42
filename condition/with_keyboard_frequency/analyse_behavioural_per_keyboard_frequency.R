library(dplyr)
library(tidyr)
library(ggplot2)
library(ez)
library(car)
library(readr)

# Input/output setup
outcome_variable <- "keyboard_frequency"
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
  P_levene_value = numeric(),
  stringsAsFactors = FALSE
)

# Loop over variables
for (var_name in variable_names) {
  subset_data <- data %>% filter(name == var_name)
  
  # Ensure participant and hours are factors
  subset_data <- subset_data %>%
    mutate(participant = as.factor(participant),
           keyboard_frequency = as.factor(keyboard_frequency))
  
  # Aggregate: mean per participant × keyboard_frequency
  aggregated <- subset_data %>%
    group_by(participant, keyboard_frequency) %>%
    summarise(value = mean(value), .groups = "drop")
  
  if (nrow(aggregated) > 0) {
    levene_result <- leveneTest(value ~ keyboard_frequency, data = aggregated)
    levene_p <- levene_result$`Pr(>F)`[1]
    cat("Variable:", var_name, "\n")
    cat("Levene's Test p-value:", levene_p, "\n")
    
    # Run ANOVA
    anova_result <- tryCatch({
      model <- aov(value ~ keyboard_frequency, data = aggregated)
      summary_model <- summary(model)[[1]]
      
      F_val <- summary_model$`F value`[1]
      Df_num <- summary_model$Df[1]
      Df_den <- summary_model$Df[2]
      P_val <- summary_model$`Pr(>F)`[1]
      
      # Summary stats
      summary_data <- aggregated %>%
        group_by(keyboard_frequency) %>%
        summarise(mean = mean(value),
                  se = sd(value) / sqrt(n()),
                  .groups = "drop") %>%
        pivot_wider(
          names_from = keyboard_frequency,
          values_from = c(mean, se),
          names_glue = "{keyboard_frequency}_{.value}"
        )
      # Pairwise t-tests (only if ANOVA is significant)
      pairwise_text <- NA
      if (P_val < 0.05) {
        pw <- pairwise.t.test(
          x = aggregated$value,
          g = aggregated$keyboard_frequency,
          paired = FALSE,
          p.adjust.method = "bonferroni"
        )
        
        # Capture significant comparisons
        sig_pairs <- which(pw$p.value < 0.05, arr.ind = TRUE)
        sig_labels <- if (nrow(sig_pairs) > 0) {
          apply(sig_pairs, 1, function(i) {
            paste0(rownames(pw$p.value)[i[1]], " vs ", colnames(pw$p.value)[i[2]], 
                   " (p=", signif(pw$p.value[i[1], i[2]], 3), ")")
          })
        } else {
          "None"
        }
        
        pairwise_text <- paste(sig_labels, collapse = "; ")
      }
      
      # Combine results
      combined <- cbind(data.frame(
        Name = var_name,
        F_value = F_val,
        Df_num = Df_num,
        Df_den = Df_den,
        P_value = P_val,
        P_levene_value = levene_p,
        Significant_Pairs = pairwise_text
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
