# =============================================================================
# Bayesian Hurdle-Gamma Analysis: Glutamate & GABA in Microglia
#
# Purpose: Fit Bayesian hurdle-gamma models for immunogold GABA and glutamate
#          particle density in microglia (Control vs LPS), and generate
#          publication-ready violin plots with posterior statistics.
#
# Manuscript figures: Fig 1C,F and Fig 5C,F
#
# Input:  profile_level_data.csv (immunogold particle densities per profile)
# Output: hurdle_glutamate_gaba_violin.png (main side-by-side violin)
#         hurdle_grouped_violin.png (grouped alternative)
#         hurdle_raincloud.png (raincloud alternative)
#         hurdle_summary_stats.csv
#         hurdle_glutamate_model.rds / hurdle_gaba_model.rds
# =============================================================================

library(brms)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

set.seed(42)
options(mc.cores = 14)
output_dir <- "/Users/cjsogn/Bayes-microglia"
dir.create(output_dir, showWarnings = FALSE)

# =============================================================================
# Load and Prepare Data
# =============================================================================
data <- read.csv("/Users/cjsogn/microglia_glutamate_gaba_analysis/results/profile_level_data.csv")

# Filter microglia data for both markers
microglia_data <- data %>%
  filter(Compartment == "Microglia") %>%
  mutate(
    Treatment = factor(Treatment, levels = c("Control", "LPS")),
    Marker = factor(Antibody, levels = c("GLUT", "GABA"),
                    labels = c("Glutamate", "GABA"))
  ) %>%
  filter(!is.na(Density))

cat("=== Data Summary ===\n")
cat("Total observations:", nrow(microglia_data), "\n")
cat("N animals:", length(unique(microglia_data$Animal)), "\n\n")

data_summary <- microglia_data %>%
  group_by(Marker, Treatment) %>%
  summarise(
    n = n(),
    n_zeros = sum(Density == 0),
    pct_zeros = round(mean(Density == 0) * 100, 1),
    mean_density = round(mean(Density), 2),
    sd_density = round(sd(Density), 2),
    .groups = "drop"
  )
print(data_summary)

# =============================================================================
# Fit Hurdle-Gamma Models
# =============================================================================

# Glutamate model
cat("\n=== Fitting Glutamate Model ===\n")
glut_data <- microglia_data %>% filter(Marker == "Glutamate")

hurdle_glut <- brm(
  bf(
    Density ~ Treatment + (1 | Animal),
    hu ~ Treatment
  ),
  family = hurdle_gamma(link = "log"),
  data = glut_data,
  chains = 4,
  iter = 5000,
  warmup = 1000,
  cores = 14,
  seed = 42,
  silent = 2,
  refresh = 0,
  control = list(adapt_delta = 0.999, max_treedepth = 15)
)

# GABA model
cat("\n=== Fitting GABA Model ===\n")
gaba_data <- microglia_data %>% filter(Marker == "GABA")

hurdle_gaba <- brm(
  bf(
    Density ~ Treatment + (1 | Animal),
    hu ~ Treatment
  ),
  family = hurdle_gamma(link = "log"),
  data = gaba_data,
  chains = 4,
  iter = 5000,
  warmup = 1000,
  cores = 14,
  seed = 42,
  silent = 2,
  refresh = 0,
  control = list(adapt_delta = 0.999, max_treedepth = 15)
)

# =============================================================================
# Extract Posterior Statistics
# =============================================================================

extract_results <- function(model, marker_name) {
  posts <- as_draws_df(model)

  # Effect on non-zero density (log scale)
  prob_reduces <- mean(posts$b_TreatmentLPS < 0)
  prob_increases_zeros <- mean(posts$b_hu_TreatmentLPS > 0)

  # Multiplicative effect
  mult_effect <- exp(posts$b_TreatmentLPS)
  pct_change <- (mult_effect - 1) * 100

  # Predicted means
  ctrl_mean <- exp(posts$b_Intercept)
  lps_mean <- exp(posts$b_Intercept + posts$b_TreatmentLPS)

  list(
    marker = marker_name,
    prob_reduces = prob_reduces,
    prob_increases_zeros = prob_increases_zeros,
    pct_change_median = median(pct_change),
    pct_change_ci = quantile(pct_change, c(0.025, 0.975)),
    ctrl_mean = median(ctrl_mean),
    ctrl_ci = quantile(ctrl_mean, c(0.025, 0.975)),
    lps_mean = median(lps_mean),
    lps_ci = quantile(lps_mean, c(0.025, 0.975)),
    posterior_samples = posts
  )
}

results_glut <- extract_results(hurdle_glut, "Glutamate")
results_gaba <- extract_results(hurdle_gaba, "GABA")

# Print results
cat("\n=== Glutamate Results ===\n")
cat("P(LPS reduces density):", round(results_glut$prob_reduces, 4), "\n")
cat("Percent change:", round(results_glut$pct_change_median, 1), "% [",
    round(results_glut$pct_change_ci[1], 1), ",",
    round(results_glut$pct_change_ci[2], 1), "]\n")

cat("\n=== GABA Results ===\n")
cat("P(LPS reduces density):", round(results_gaba$prob_reduces, 4), "\n")
cat("Percent change:", round(results_gaba$pct_change_median, 1), "% [",
    round(results_gaba$pct_change_ci[1], 1), ",",
    round(results_gaba$pct_change_ci[2], 1), "]\n")

# =============================================================================
# Create Publication-Ready Violin Plots
# =============================================================================

# Color palettes (Nature/Science style - accessible)
# GABA: shades of red
gaba_colors <- c("Control" = "#EF5350", "LPS" = "#C62828")
gaba_fill_colors <- c("Control" = "#FFCDD2", "LPS" = "#EF9A9A")

# Glutamate: shades of blue
glut_colors <- c("Control" = "#64B5F6", "LPS" = "#1565C0")
glut_fill_colors <- c("Control" = "#BBDEFB", "LPS" = "#90CAF9")

# Function to generate significance annotation
get_sig_label <- function(prob_reduces) {
  if (prob_reduces >= 0.999) return("***")
  if (prob_reduces >= 0.99) return("**")
  if (prob_reduces >= 0.95) return("*")
  return("n.s.")
}

# Calculate animal means for overlay
animal_means <- microglia_data %>%
  group_by(Animal, Treatment, Marker) %>%
  summarise(Mean = mean(Density, na.rm = TRUE), .groups = "drop")

# Get y-axis limits for each marker
get_y_max <- function(marker) {
  max(microglia_data$Density[microglia_data$Marker == marker], na.rm = TRUE) * 1.2
}

# Create individual violin plots
create_violin <- function(marker_name, results, y_max = NULL, colors, fill_colors) {

  plot_data <- microglia_data %>% filter(Marker == marker_name)
  animal_data <- animal_means %>% filter(Marker == marker_name)

  if (is.null(y_max)) y_max <- get_y_max(marker_name)

  # Significance
  sig_label <- get_sig_label(results$prob_reduces)
  prob_text <- sprintf("P = %.3f", results$prob_reduces)

  # Model predictions
  pred_df <- data.frame(
    Treatment = factor(c("Control", "LPS"), levels = c("Control", "LPS")),
    Mean = c(results$ctrl_mean, results$lps_mean),
    Lower = c(results$ctrl_ci[1], results$lps_ci[1]),
    Upper = c(results$ctrl_ci[2], results$lps_ci[2])
  )

  p <- ggplot(plot_data, aes(x = Treatment, y = Density)) +
    # Violin
    geom_violin(aes(fill = Treatment),
                alpha = 0.65,
                width = 0.85,
                trim = TRUE,
                color = NA,
                scale = "width",
                linewidth = 0.4) +
    # Outline
    geom_violin(aes(color = Treatment),
                fill = NA,
                width = 0.85,
                trim = TRUE,
                scale = "width",
                linewidth = 0.5) +
    # Box plot inside
    geom_boxplot(width = 0.12,
                 outlier.shape = NA,
                 fill = "white",
                 color = "gray30",
                 alpha = 0.9,
                 linewidth = 0.4) +
    # Animal means as diamonds
    geom_point(data = animal_data,
               aes(x = Treatment, y = Mean, fill = Treatment),
               shape = 23, size = 2.5, color = "black", stroke = 0.6,
               position = position_jitter(width = 0.08, seed = 42)) +
    # Model posterior means with CIs
    geom_pointrange(data = pred_df,
                    aes(x = Treatment, y = Mean, ymin = Lower, ymax = Upper),
                    color = "black", linewidth = 0.9, size = 0.5, fatten = 2.5) +
    # Significance bracket
    annotate("segment", x = 1, xend = 2, y = y_max * 0.86, yend = y_max * 0.86,
             linewidth = 0.6, color = "black") +
    annotate("segment", x = 1, xend = 1, y = y_max * 0.83, yend = y_max * 0.86,
             linewidth = 0.6, color = "black") +
    annotate("segment", x = 2, xend = 2, y = y_max * 0.83, yend = y_max * 0.86,
             linewidth = 0.6, color = "black") +
    # Significance label
    annotate("text", x = 1.5, y = y_max * 0.91,
             label = sig_label, size = 7, fontface = "bold") +
    # Probability annotation
    annotate("text", x = 1.5, y = y_max * 0.98,
             label = prob_text, size = 3.2, color = "gray35") +
    # Scales
    scale_fill_manual(values = fill_colors) +
    scale_color_manual(values = colors) +
    scale_y_continuous(limits = c(0, y_max), expand = c(0, 0),
                       breaks = scales::pretty_breaks(n = 5)) +
    # Labels
    labs(
      x = NULL,
      y = expression("Particle Density (particles/"*mu*"m"^2*")"),
      title = marker_name
    ) +
    # Theme
    theme_classic(base_size = 11) +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5,
                                margin = margin(b = 8)),
      axis.title.y = element_text(size = 10, margin = margin(r = 8)),
      axis.text.x = element_text(size = 11, color = "black", face = "bold"),
      axis.text.y = element_text(size = 9, color = "black"),
      axis.line = element_line(linewidth = 0.5, color = "black"),
      axis.ticks = element_line(linewidth = 0.4, color = "black"),
      axis.ticks.length = unit(0.15, "cm"),
      panel.grid = element_blank(),
      plot.margin = margin(12, 12, 8, 8)
    )

  return(p)
}

# Create both violin plots with matched y-axis
y_max_glut <- get_y_max("Glutamate")
y_max_gaba <- get_y_max("GABA")
y_max_shared <- max(y_max_glut, y_max_gaba)

p_glut <- create_violin("Glutamate", results_glut, y_max_shared, glut_colors, glut_fill_colors)
p_gaba <- create_violin("GABA", results_gaba, y_max_shared, gaba_colors, gaba_fill_colors)

# =============================================================================
# Combined Violin Plot Figure
# =============================================================================

combined_violin <- p_glut + p_gaba +
  plot_annotation(
    title = "Effect of LPS on Neurotransmitter Content in Microglia",
    subtitle = "Bayesian Hurdle-Gamma Model",
    caption = expression(
      "Diamonds: individual animal means | Black points: posterior mean " %+-% " 95% CI | Significance: * P > 0.95, ** P > 0.99, *** P > 0.999"
    ),
    tag_levels = "A",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40",
                                   margin = margin(b = 12)),
      plot.caption = element_text(size = 8, hjust = 0.5, color = "gray45",
                                  margin = margin(t = 10)),
      plot.tag = element_text(size = 12, face = "bold")
    )
  )

ggsave(file.path(output_dir, "hurdle_glutamate_gaba_violin.png"), combined_violin,
       width = 8, height = 5.5, dpi = 300, bg = "white")

# =============================================================================
# Alternative: Grouped Bar-Style Violin (Side-by-Side within Marker)
# =============================================================================

# Create combined data with model predictions for both markers
pred_combined <- data.frame(
  Treatment = factor(rep(c("Control", "LPS"), 2), levels = c("Control", "LPS")),
  Marker = factor(rep(c("Glutamate", "GABA"), each = 2), levels = c("Glutamate", "GABA")),
  Mean = c(results_glut$ctrl_mean, results_glut$lps_mean,
           results_gaba$ctrl_mean, results_gaba$lps_mean),
  Lower = c(results_glut$ctrl_ci[1], results_glut$lps_ci[1],
            results_gaba$ctrl_ci[1], results_gaba$lps_ci[1]),
  Upper = c(results_glut$ctrl_ci[2], results_glut$lps_ci[2],
            results_gaba$ctrl_ci[2], results_gaba$lps_ci[2])
)

# For grouped plots, create combined color scales
grouped_fill <- c("Control" = "#BBDEFB", "LPS" = "#90CAF9")  # Blue shades
grouped_colors <- c("Control" = "#64B5F6", "LPS" = "#1565C0")

grouped_violin <- ggplot(microglia_data, aes(x = Marker, y = Density, fill = Treatment)) +
  geom_violin(position = position_dodge(width = 0.8),
              alpha = 0.65,
              width = 0.7,
              trim = TRUE,
              scale = "width",
              color = NA) +
  geom_violin(aes(color = Treatment),
              position = position_dodge(width = 0.8),
              fill = NA,
              width = 0.7,
              trim = TRUE,
              scale = "width",
              linewidth = 0.5) +
  geom_boxplot(position = position_dodge(width = 0.8),
               width = 0.1,
               outlier.shape = NA,
               fill = "white",
               color = "gray30",
               alpha = 0.9,
               linewidth = 0.35) +
  # Model predictions
  geom_pointrange(data = pred_combined,
                  aes(x = Marker, y = Mean, ymin = Lower, ymax = Upper,
                      group = Treatment),
                  position = position_dodge(width = 0.8),
                  color = "black", linewidth = 0.7, size = 0.4, fatten = 2) +
  # Significance brackets - Glutamate
  annotate("segment", x = 0.8, xend = 1.2, y = y_max_shared * 0.88, yend = y_max_shared * 0.88,
           linewidth = 0.5) +
  annotate("segment", x = 0.8, xend = 0.8, y = y_max_shared * 0.85, yend = y_max_shared * 0.88,
           linewidth = 0.5) +
  annotate("segment", x = 1.2, xend = 1.2, y = y_max_shared * 0.85, yend = y_max_shared * 0.88,
           linewidth = 0.5) +
  annotate("text", x = 1, y = y_max_shared * 0.93,
           label = get_sig_label(results_glut$prob_reduces), size = 5.5, fontface = "bold") +
  # Significance brackets - GABA
  annotate("segment", x = 1.8, xend = 2.2, y = y_max_shared * 0.88, yend = y_max_shared * 0.88,
           linewidth = 0.5) +
  annotate("segment", x = 1.8, xend = 1.8, y = y_max_shared * 0.85, yend = y_max_shared * 0.88,
           linewidth = 0.5) +
  annotate("segment", x = 2.2, xend = 2.2, y = y_max_shared * 0.85, yend = y_max_shared * 0.88,
           linewidth = 0.5) +
  annotate("text", x = 2, y = y_max_shared * 0.93,
           label = get_sig_label(results_gaba$prob_reduces), size = 5.5, fontface = "bold") +
  scale_fill_manual(values = grouped_fill, name = "Treatment") +
  scale_color_manual(values = grouped_colors, guide = "none") +
  scale_y_continuous(limits = c(0, y_max_shared), expand = c(0, 0),
                     breaks = scales::pretty_breaks(n = 5)) +
  labs(
    x = NULL,
    y = expression("Particle Density (particles/"*mu*"m"^2*")"),
    title = "Neurotransmitter Content in Microglia",
    subtitle = "Control vs LPS Treatment"
  ) +
  theme_classic(base_size = 11) +
  theme(
    legend.position = c(0.92, 0.88),
    legend.background = element_rect(fill = "white", color = NA),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9),
    legend.key.size = unit(0.4, "cm"),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40",
                                 margin = margin(b = 12)),
    axis.title.y = element_text(size = 10, margin = margin(r = 8)),
    axis.text.x = element_text(size = 11, color = "black", face = "bold"),
    axis.text.y = element_text(size = 9, color = "black"),
    axis.line = element_line(linewidth = 0.5),
    axis.ticks = element_line(linewidth = 0.4),
    axis.ticks.length = unit(0.15, "cm"),
    panel.grid = element_blank(),
    plot.margin = margin(12, 15, 10, 10)
  )

ggsave(file.path(output_dir, "hurdle_grouped_violin.png"), grouped_violin,
       width = 6, height = 5.5, dpi = 300, bg = "white")

# =============================================================================
# Raincloud Plot (Modern Alternative)
# =============================================================================

# Create offset positions for raincloud
set.seed(42)
microglia_jitter <- microglia_data %>%
  mutate(
    x_base = as.numeric(factor(Marker)),
    x_offset = ifelse(Treatment == "Control", -0.15, 0.15),
    x_jitter = x_base + x_offset + runif(n(), -0.05, 0.05)
  )

raincloud <- ggplot(microglia_data, aes(x = Marker, y = Density, fill = Treatment)) +
  # Half violins (rotated)
  geom_violin(aes(group = interaction(Marker, Treatment)),
              position = position_dodge(width = 0.7),
              alpha = 0.6,
              width = 0.6,
              trim = TRUE,
              scale = "width",
              color = NA) +
  geom_violin(aes(color = Treatment, group = interaction(Marker, Treatment)),
              position = position_dodge(width = 0.7),
              fill = NA,
              width = 0.6,
              trim = TRUE,
              scale = "width",
              linewidth = 0.5) +
  # Boxplot
  geom_boxplot(position = position_dodge(width = 0.7),
               width = 0.12,
               outlier.shape = NA,
               fill = "white",
               color = "gray30",
               alpha = 0.95,
               linewidth = 0.4) +
  # Model predictions
  geom_pointrange(data = pred_combined,
                  aes(x = Marker, y = Mean, ymin = Lower, ymax = Upper,
                      group = Treatment),
                  position = position_dodge(width = 0.7),
                  color = "black", linewidth = 0.8, size = 0.45, fatten = 2.2) +
  # Significance brackets
  annotate("segment", x = 0.82, xend = 1.18, y = y_max_shared * 0.88, yend = y_max_shared * 0.88,
           linewidth = 0.5) +
  annotate("text", x = 1, y = y_max_shared * 0.93,
           label = get_sig_label(results_glut$prob_reduces), size = 5, fontface = "bold") +
  annotate("segment", x = 1.82, xend = 2.18, y = y_max_shared * 0.88, yend = y_max_shared * 0.88,
           linewidth = 0.5) +
  annotate("text", x = 2, y = y_max_shared * 0.93,
           label = get_sig_label(results_gaba$prob_reduces), size = 5, fontface = "bold") +
  scale_fill_manual(values = grouped_fill, name = "Treatment") +
  scale_color_manual(values = grouped_colors, guide = "none") +
  scale_y_continuous(limits = c(0, y_max_shared), expand = c(0, 0)) +
  labs(
    x = NULL,
    y = expression("Particle Density (particles/"*mu*"m"^2*")"),
    title = "Neurotransmitter Content in Microglia",
    subtitle = "Bayesian Hurdle-Gamma Model | Control vs LPS"
  ) +
  theme_classic(base_size = 11) +
  theme(
    legend.position = "top",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40",
                                 margin = margin(b = 8)),
    axis.title.y = element_text(size = 10, margin = margin(r = 8)),
    axis.text.x = element_text(size = 11, color = "black", face = "bold"),
    axis.text.y = element_text(size = 9, color = "black"),
    axis.line = element_line(linewidth = 0.5),
    panel.grid = element_blank(),
    plot.margin = margin(8, 15, 10, 10)
  )

ggsave(file.path(output_dir, "hurdle_raincloud.png"), raincloud,
       width = 6, height = 5.5, dpi = 300, bg = "white")

# =============================================================================
# Summary Statistics Table
# =============================================================================

summary_table <- data.frame(
  Marker = c("Glutamate", "GABA"),
  `Control Mean (95% CI)` = c(
    sprintf("%.1f (%.1f-%.1f)", results_glut$ctrl_mean,
            results_glut$ctrl_ci[1], results_glut$ctrl_ci[2]),
    sprintf("%.1f (%.1f-%.1f)", results_gaba$ctrl_mean,
            results_gaba$ctrl_ci[1], results_gaba$ctrl_ci[2])
  ),
  `LPS Mean (95% CI)` = c(
    sprintf("%.1f (%.1f-%.1f)", results_glut$lps_mean,
            results_glut$lps_ci[1], results_glut$lps_ci[2]),
    sprintf("%.1f (%.1f-%.1f)", results_gaba$lps_mean,
            results_gaba$lps_ci[1], results_gaba$lps_ci[2])
  ),
  `% Change (95% CI)` = c(
    sprintf("%.1f%% (%.1f to %.1f%%)", results_glut$pct_change_median,
            results_glut$pct_change_ci[1], results_glut$pct_change_ci[2]),
    sprintf("%.1f%% (%.1f to %.1f%%)", results_gaba$pct_change_median,
            results_gaba$pct_change_ci[1], results_gaba$pct_change_ci[2])
  ),
  `P(LPS reduces)` = c(
    sprintf("%.4f", results_glut$prob_reduces),
    sprintf("%.4f", results_gaba$prob_reduces)
  ),
  check.names = FALSE
)

cat("\n=== Summary Statistics ===\n")
print(summary_table, row.names = FALSE)

write.csv(summary_table, file.path(output_dir, "hurdle_summary_stats.csv"),
          row.names = FALSE)

# =============================================================================
# Save Models
# =============================================================================
saveRDS(hurdle_glut, file.path(output_dir, "hurdle_glutamate_model.rds"))
saveRDS(hurdle_gaba, file.path(output_dir, "hurdle_gaba_model.rds"))

cat("\n=== Output saved to", output_dir, "===\n")
cat("- hurdle_glutamate_gaba_violin.png (main figure - side by side)\n")
cat("- hurdle_grouped_violin.png (alternative - grouped)\n")
cat("- hurdle_raincloud.png (modern raincloud style)\n")
cat("- hurdle_summary_stats.csv\n")
cat("- hurdle_glutamate_model.rds\n")
cat("- hurdle_gaba_model.rds\n")
