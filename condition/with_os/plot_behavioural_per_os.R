library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(forcats)

# Load ANOVA results
results <- read_csv("processed_data/behavioural/42-operating_system-output-os.csv")

# Significance stars function
get_significance <- function(p) {
  if (p < 0.001) return("***")
  if (p < 0.01) return("**")
  if (p < 0.05) return("*")
  return("")
}

# Mapping variable names to descriptive labels with units
name_labels <- list(
  mouse_toolbar_navigation_efficiency = "Mouse navigation efficiency on the toolbar (%)",
  mouse_confirm_dialog_duration = "Time taken to confirm the popup (s)",
  mouse_toolbar_navigation_speed = "Mouse navigation speed on the toolbar (px/s)",
  keyboard_pressed_duration = "Duration of key pressed on the keyboard (ms)",
  mouse_close_window_duration = "Time taken to close the application (s)"
)

# Filter for significant results only
sig_results <- results %>%
  filter(P_value < 0.05) %>%
  mutate(
    Significance = sapply(P_value, get_significance),
    Label = sapply(Name, function(x) ifelse(!is.null(name_labels[[x]]), name_labels[[x]], x))
  )

for (i in seq_len(nrow(sig_results))) {
  row <- sig_results[i, ]
  
  plot_data <- data.frame(
    Group = factor(c("Windows", "macOS"), levels = c("Windows", "macOS")),
    Mean = c(row$windows_mean, row$mac_mean),
    SE = c(row$windows_se, row$mac_se)
  )
  
  y1 <- 1
  y2 <- 2
  
  max_x <- max(plot_data$Mean + plot_data$SE)
  bracket_x <- max_x * 1.1   # bracket just right of max bar end
  
  star_x <- bracket_x + 0.0 # stars & p-value a bit further right
  star_y <- mean(c(y1, y2))  # vertically centered between bars
  
  p_val_text <- if (row$P_value < 0.001) {
    "p < 0.001"
  } else {
    sprintf("p = %.3f", row$P_value)
  }
  
  p <- ggplot(plot_data, aes(x = Mean, y = Group, fill = Group)) +
    geom_bar(stat = "identity", width = 0.6) +
    geom_errorbar(aes(xmin = Mean - SE, xmax = Mean + SE), width = 0.2) +
    
    # vertical bracket line connecting bars
    annotate("segment", x = bracket_x, xend = bracket_x, y = y1, yend = y2, size = 0.8) +
    
    # horizontal ticks at top and bottom of bracket
    annotate("segment", x = bracket_x, xend = bracket_x - 0.03*max_x, y = y1, yend = y1, size = 0.8) +
    annotate("segment", x = bracket_x, xend = bracket_x - 0.03*max_x, y = y2, yend = y2, size = 0.8) +
    
    # significance stars with padding label, a bit above center
    annotate("label", x = star_x, y = star_y + 0.04, label = row$Significance,
             size = 8, fontface = "bold", fill = "white", label.size = 0) +
    
    # p-value label with padding, a bit below center
    annotate("label", x = star_x, y = star_y - 0.08, label = p_val_text,
             size = 5, fill = "white", label.size = 0) +
    
    scale_fill_manual(values = c("Windows" = "#1E88E5",  # Blue (strong, tech-associated, and accessible)
                                 "macOS"  = "#D81B60")   # Pink-red (matches macOS accent style)
                      ) +
    labs(
      title = row$Label,
      x = "Mean ± SE",   # nice ± symbol for x-axis label
      y = NULL,
      fill = NULL
    ) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "none",
      panel.grid.major.y = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.margin = margin(1, 2, 1, 1, "cm")  # more right margin for bracket & labels
    ) +
    coord_cartesian(xlim = c(0, bracket_x + 0.2*max_x), clip = "off")
  
  print(p)
}
