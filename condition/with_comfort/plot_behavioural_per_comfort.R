library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(forcats)

# Load ANOVA results
results <- read_csv("processed_data/behavioural/42-comfort-output.csv")

# Significance stars function
get_significance <- function(p) {
  if (p < 0.001) return("***")
  if (p < 0.01) return("**")
  if (p < 0.05) return("*")
  return("")
}

# Mapping variable names to descriptive labels with units
name_labels <- list(
  mouse_open_trash_bin_duration = "Time taken to open trash bin app (s)",
  mouse_open_notes_duration = "Time taken to open notes app (s)",
  mouse_grouped_selection_duration = "Time taken to select a group of files (s)",
  mouse_confirm_dialog_duration = "Time taken to confirm the popup (s)",
  mouse_selection_coverage = "Coverage of selected area (%)",
  mouse_toolbar_navigation_speed = "Mouse navigation speed on the toolbar (px/s)"
)

# Filter for significant results only
sig_results <- results %>%
  filter(P_value < 0.05) %>%
  mutate(
    Significance = sapply(P_value, get_significance),
    Label = sapply(Name, function(x) ifelse(!is.null(name_labels[[x]]), name_labels[[x]], x))
  )

# Iterate over each significant variable
for (i in seq_len(nrow(sig_results))) {
  row <- sig_results[i, ]
  
  # Prepare data for bar plot
  plot_data <- data.frame(
    Group = factor(c("Very Comfortable", "Comfortable", "Neutral"), 
                   levels = c("Very Comfortable", "Comfortable", "Neutral")),
    Mean = c(row$`1_mean`, row$`2_mean`, row$`3_mean`),
    SE = c(row$`1_se`, row$`2_se`, row$`3_se`)
  )
  
  # Prepare max x for layout
  max_x <- max(plot_data$Mean + plot_data$SE, na.rm = TRUE)
  
  # Base plot
  p <- ggplot(plot_data, aes(x = Mean, y = Group, fill = Group)) +
    geom_bar(stat = "identity", width = 0.6) +
    geom_errorbar(aes(xmin = Mean - SE, xmax = Mean + SE), width = 0.2) +
    scale_fill_manual(values = c(
      "Very Comfortable" = "#1E88E5",
      "Comfortable" = "#FFC107",
      "Neutral" = "#D81B60"
    )) +
    labs(
      title = row$Label,
      x = "Mean ± SE",
      y = NULL,
      fill = NULL
    ) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "none",
      panel.grid.major.y = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.margin = margin(1, 3, 1, 1, "cm")
    ) +
    coord_cartesian(xlim = c(0, max_x * 1.5), clip = "off")
  
  # Add significant pair brackets if present
  pair_string <- row$Significant_Pairs
  if (!is.na(pair_string)) {
    pairs <- str_split(pair_string, ";\\s*")[[1]]
    y_levels <- levels(plot_data$Group)
    y_indices <- setNames(1:length(y_levels), y_levels)
    offset_index <- 0
    
    for (pair in pairs) {
      m <- str_match(pair, "(\\d+) vs (\\d+) \\(p=([0-9.]+)\\)")
      if (!is.na(m[1,1])) {
        group1 <- as.integer(m[1,2])
        group2 <- as.integer(m[1,3])
        p_val <- as.numeric(m[1,4])
        
        # Convert numeric to labels
        label1 <- as.character(plot_data$Group[group1])
        label2 <- as.character(plot_data$Group[group2])
        y1 <- y_indices[[label1]]
        y2 <- y_indices[[label2]]
        y_mid <- mean(c(y1, y2)) + offset_index * 0.2
        
        # Format stars and p-value text
        stars <- get_significance(p_val)
        p_text <- if (p_val < 0.001) "p < 0.001" else sprintf("p = %.3f", p_val)
        
        # Position brackets and text
        bracket_x <- max_x * 1.1 + offset_index * 0.3 * max_x
        star_x <- bracket_x + 0.03 * max_x
        
        # Draw bracket
        p <- p +
          annotate("segment", x = bracket_x, xend = bracket_x, y = y1, yend = y2, size = 0.8) +
          annotate("segment", x = bracket_x, xend = bracket_x - 0.03*max_x, y = y1, yend = y1, size = 0.8) +
          annotate("segment", x = bracket_x, xend = bracket_x - 0.03*max_x, y = y2, yend = y2, size = 0.8) +
          annotate("label", x = star_x, y = y_mid + 0.08, label = stars,
                   size = 6, fontface = "bold", fill = "white", label.size = 0) +
          annotate("label", x = star_x, y = y_mid - 0.05, label = p_text,
                   size = 4.5, fill = "white", label.size = 0)
        
        offset_index <- offset_index + 1
      }
    }
  }
  
  print(p)
}
