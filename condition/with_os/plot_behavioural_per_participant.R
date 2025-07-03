library(dplyr)
library(tidyr)
library(ggplot2)
library(ez)
library(readr)

# Input setup
outcome_variable <- "operating_system"
data <- read.csv(paste0("processed_data/behavioural/42-", outcome_variable, ".csv"))

# Filter and plot for each name + native OS combo
plots_data <- list()

for (native_os in c("windows", "mac")) {
  variable_names <- c("mouse_toolbar_navigation_efficiency", "mouse_confirm_dialog_duration", "mouse_toolbar_navigation_speed", "mouse_close_window_duration")
  
  for (var_name in variable_names) {
    subset_data <- data %>% filter(name == var_name)
    
    # Ensure factors
    subset_data$participant <- as.factor(subset_data$participant)
    subset_data$preferred <- as.factor(subset_data$preferred)
    
    # Identify suitable participants
    participant_prefers_os <- subset_data %>%
      filter(os == native_os, preferred == "True") %>%
      distinct(participant)
    
    participant_os_counts <- subset_data %>%
      group_by(participant) %>%
      summarise(n_os_options = n_distinct(preferred), .groups = "drop") %>%
      filter(n_os_options > 1)
    
    suitable_participants <- intersect(participant_os_counts$participant, participant_prefers_os$participant)
    
    subset_data <- subset_data %>%
      filter(participant %in% suitable_participants)
    
    if (nrow(subset_data) > 0 && length(unique(subset_data$preferred)) > 1) {
      # Prepare data for plotting
      plot_df <- subset_data %>%
        group_by(participant, preferred) %>%
        summarise(mean_value = mean(value), .groups = "drop") %>%
        mutate(Name = var_name,
               Native = native_os)
      
      plots_data[[length(plots_data) + 1]] <- plot_df
    }
  }
}

# Combine all for a single plot
all_plot_data <- bind_rows(plots_data)

# Plot: bar chart per participant with preferred conditions, faceted by variable name
ggplot(all_plot_data, aes(x = participant, y = mean_value, fill = preferred)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  facet_wrap(~ Name + Native, scales = "free_x") +
  labs(title = "Mean Value by Preferred OS and Participant",
       x = "Participant", y = "Mean Value",
       fill = "Preferred") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
