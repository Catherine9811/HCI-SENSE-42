library(dplyr)
library(tidyr)
library(ggplot2)
library(ez)
library(readr)

# Input and output setup (same as your original structure)
outcome_variable <- "operating_system"
data <- read.csv(paste("processed_data/behavioural/42-", outcome_variable, ".csv", sep=""), sep=",")
output_file <- paste("processed_data/behavioural/42-", outcome_variable, "-output-os.csv", sep="")



# Step 1: Calculate counts and percentages for each combination of preferred and os
participant_os_counts <- data %>%
  group_by(participant) %>%
  summarise(n_os_options = n_distinct(preferred)) %>%
  filter(n_os_options > 1)

os_labels <- c(
  windows = "Windows",
  mac = "macOS"
)

os_summary <- data %>%
  filter(participant %in% participant_os_counts$participant) %>%
  filter(preferred == "True") %>%
  distinct(participant, os) %>%
  count(os) %>%
  mutate(percentage = n / sum(n) * 100,
         label = dplyr::recode(os, !!!os_labels))

# Step 2: Create a pie chart using ggplot2
ggplot(os_summary, aes(x = "", y = percentage, fill = label)) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  labs(title = "Percentage of Preference for Operating System",
       fill = "Operating System") +
  theme_void() +  # clean theme without axis
  theme(legend.position = "right") +
  geom_text(aes(label = paste0(round(percentage, 1), "% (", n, ")")),
            position = position_stack(vjust = 0.5))


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

# Loop through each variable name
for (var_name in variable_names) {
  subset_data <- data %>% filter(name == var_name)
  
  # Ensure participant and os are treated as factors
  # subset_data$participant <- as.factor(subset_data$participant)
  subset_data$os <- as.factor(subset_data$os)
  
  participant_os_counts <- subset_data %>%
    group_by(participant) %>%
    summarise(n_os_options = n_distinct(preferred)) %>%
    filter(n_os_options > 1)
  
  subset_data <- subset_data %>%
    filter(participant %in% participant_os_counts$participant)
  
  subset_data <- subset_data %>%
    group_by(participant, os) %>%
    summarise(value = mean(value, na.rm = TRUE), .groups = "drop")
  
  subset_data$participant <- as.factor(subset_data$participant)
  
  if (nrow(subset_data) > 0 && length(unique(subset_data$os)) > 1) {
    # ezANOVA requires one row per condition per participant
    anova_result <- tryCatch({
      ez_result <- ezANOVA(
        data = subset_data,
        dv = value,
        wid = participant,
        within = .(os),
        return_aov = TRUE,
        type = 3
      )
      # Mauchly's sphericity test holds true for os with 2 levels
      ez_main <- ez_result$ANOVA[1, ]
      
      summary_data <- subset_data %>%
        group_by(os) %>%
        summarise(mean = mean(value),
                  se = sd(value) / sqrt(n()),
                  .groups = "drop")
      flat_summary <- summary_data %>%
        pivot_wider(
          names_from = os,
          values_from = c(mean, se),
          names_glue = "{os}_{.value}"
        )
      
      combined <- cbind(data.frame(
        Name = var_name,
        F_value = ez_main$F,
        Df_num = ez_main$DFn,
        Df_den = ez_main$DFd,
        P_value = ez_main$p,
        stringsAsFactors = FALSE
      ), flat_summary)
      
      results <- bind_rows(results, combined)
    }, error = function(e) {
      # Skip and continue if ANOVA fails
      message(glue::glue("ANOVA failed for variable: {var_name}"))
    })
  }
}

# Save results
write.csv(results, output_file, row.names = FALSE)
cat("Results saved to:", output_file, "\n")
