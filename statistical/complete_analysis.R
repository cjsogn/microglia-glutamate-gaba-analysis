################################################################################
# Complete Statistical Analysis: Microglia Glutamate/GABA Immunogold Labeling
#
# Purpose: Frequentist mixed-effects models, compartment comparisons, effect
#          sizes, and publication-ready figures for immunogold particle density.
#
# Manuscript figures: Fig 1 (compartment comparisons, Control vs LPS)
#
# Input:  profile_level_data.csv, animal_level_data.csv
# Output: analysis1_paired_tests.csv, analysis1_mixed_model.csv,
#         analysis2_statistical_tests.csv, analysis2_mixed_model.csv,
#         descriptive_stats_*.csv, Figure1-4 .png/.pdf
#
# Analysis includes:
#   1. Descriptive statistics
#   2. Paired t-tests and Wilcoxon tests (animal-level)
#   3. Permutation tests
#   4. Linear mixed-effects models
#   5. Effect sizes (Hedges' g, Cohen's d)
#   6. Publication-ready figures
################################################################################

# Load libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(effectsize)
  library(cowplot)
  library(scales)
})

# Set paths
data_path <- "/Users/cjsogn/microglia_glutamate_gaba_analysis/data"
results_path <- "/Users/cjsogn/microglia_glutamate_gaba_analysis/results"
figures_path <- "/Users/cjsogn/microglia_glutamate_gaba_analysis/figures"

# Read data
profile_data <- read.csv(file.path(data_path, "profile_level_data.csv"))
animal_data <- read.csv(file.path(data_path, "animal_level_data.csv"))

cat("Data loaded successfully\n")
cat("Total profiles:", nrow(profile_data), "\n")
cat("Animals:", length(unique(profile_data$Animal)), "\n\n")

################################################################################
# ANALYSIS 1: Compartment Comparisons (Control animals only)
################################################################################

cat(strrep("=", 80), "\n")
cat("ANALYSIS 1: COMPARTMENT COMPARISONS (Control only)\n")
cat("="," repetition of 80 =s\n\n")

# Filter to control animals
ctrl_animal <- animal_data %>% filter(Treatment == "Control")
ctrl_profile <- profile_data %>% filter(Treatment == "Control")

# Function for paired analysis
paired_analysis <- function(data, antibody_type) {
  ab_data <- data %>% filter(Antibody == antibody_type)
  
  # Pivot to wide format for paired tests
  wide_data <- ab_data %>%
    select(Animal, Compartment, Mean_Density) %>%
    pivot_wider(names_from = Compartment, values_from = Mean_Density)
  
  results <- data.frame()
  
  comparisons <- list(
    c("Microglia", "GLUT_terminals"),
    c("Microglia", "GABA_terminals"),
    c("Microglia", "Spine")
  )
  
  for (comp in comparisons) {
    x <- wide_data[[comp[1]]]
    y <- wide_data[[comp[2]]]
    
    # Paired t-test
    t_result <- t.test(x, y, paired = TRUE)
    
    # Wilcoxon signed-rank test
    w_result <- tryCatch(
      wilcox.test(x, y, paired = TRUE, exact = FALSE),
      error = function(e) list(p.value = NA, statistic = NA)
    )
    
    # Cohen's d for paired samples
    d <- mean(x - y) / sd(x - y)
    
    results <- rbind(results, data.frame(
      Antibody = antibody_type,
      Comparison = paste(comp[1], "vs", comp[2]),
      Mean_1 = mean(x),
      SD_1 = sd(x),
      Mean_2 = mean(y),
      SD_2 = sd(y),
      Difference = mean(x) - mean(y),
      Pct_diff = ((mean(x) - mean(y)) / mean(y)) * 100,
      Paired_t_statistic = t_result$statistic,
      Paired_t_pvalue = t_result$p.value,
      Paired_t_CI_lower = t_result$conf.int[1],
      Paired_t_CI_upper = t_result$conf.int[2],
      Wilcoxon_pvalue = w_result$p.value,
      Cohens_d = d,
      N_pairs = length(x)
    ))
  }
  return(results)
}

# Run paired analyses
analysis1_paired <- rbind(
  paired_analysis(ctrl_animal, "GLUT"),
  paired_analysis(ctrl_animal, "GABA")
)

# Apply Bonferroni correction (3 pairwise comparisons per antibody)
analysis1_paired <- analysis1_paired %>%
  group_by(Antibody) %>%
  mutate(
    Paired_t_pvalue_bonf = pmin(Paired_t_pvalue * n(), 1),
    Wilcoxon_pvalue_bonf = pmin(Wilcoxon_pvalue * n(), 1)
  ) %>%
  ungroup()

cat("Paired tests with Bonferroni correction:\n")
for (i in seq_len(nrow(analysis1_paired))) {
  row <- analysis1_paired[i, ]
  cat(sprintf("  %s | %s: t-test p=%.4f (Bonf p=%.4f), Wilcoxon p=%.4f (Bonf p=%.4f)\n",
              row$Antibody, row$Comparison,
              row$Paired_t_pvalue, row$Paired_t_pvalue_bonf,
              row$Wilcoxon_pvalue, row$Wilcoxon_pvalue_bonf))
}
cat("\n")

# Mixed-effects models for compartment comparisons
analysis1_mixed <- data.frame()

for (ab in c("GLUT", "GABA")) {
  ab_data <- ctrl_profile %>% filter(Antibody == ab)
  ab_data$Compartment <- factor(ab_data$Compartment, 
                                 levels = c("Microglia", "GLUT_terminals", "GABA_terminals", "Spine"))
  
  # Fit mixed model
  model <- lmer(Density_per_um2 ~ Compartment + (1|Animal), data = ab_data)
  
  # Get summary with Satterthwaite df
  model_summary <- summary(model)
  coefs <- as.data.frame(coef(model_summary))
  
  # Get confidence intervals
  ci <- confint(model, parm = "beta_", method = "Wald")
  
  for (comp in c("GLUT_terminals", "GABA_terminals", "Spine")) {
    param_name <- paste0("Compartment", comp)
    if (param_name %in% rownames(coefs)) {
      analysis1_mixed <- rbind(analysis1_mixed, data.frame(
        Antibody = ab,
        Comparison = paste("Microglia vs", comp),
        Estimate = coefs[param_name, "Estimate"],
        SE = coefs[param_name, "Std. Error"],
        df = coefs[param_name, "df"],
        t_value = coefs[param_name, "t value"],
        p_value = coefs[param_name, "Pr(>|t|)"],
        CI_lower = ci[param_name, 1],
        CI_upper = ci[param_name, 2],
        N_profiles = nrow(ab_data),
        N_animals = length(unique(ab_data$Animal))
      ))
    }
  }

  # Post-hoc pairwise comparisons with Bonferroni correction via emmeans
  emm <- emmeans(model, ~ Compartment)
  posthoc <- pairs(emm, adjust = "bonferroni")
  cat(sprintf("\n  %s - emmeans post-hoc (Bonferroni):\n", ab))
  print(summary(posthoc))
}

# Apply Bonferroni correction to mixed model p-values (3 comparisons per antibody)
analysis1_mixed <- analysis1_mixed %>%
  group_by(Antibody) %>%
  mutate(p_value_bonf = pmin(p_value * n(), 1)) %>%
  ungroup()

cat("\nMixed model results with Bonferroni correction:\n")
for (i in seq_len(nrow(analysis1_mixed))) {
  row <- analysis1_mixed[i, ]
  cat(sprintf("  %s | %s: p=%.4f (Bonf p=%.4f)\n",
              row$Antibody, row$Comparison, row$p_value, row$p_value_bonf))
}

# Save Analysis 1 results
write.csv(analysis1_paired, file.path(results_path, "analysis1_paired_tests.csv"), row.names = FALSE)
write.csv(analysis1_mixed, file.path(results_path, "analysis1_mixed_model.csv"), row.names = FALSE)

cat("\nAnalysis 1 complete. Results saved.\n\n")

################################################################################
# ANALYSIS 2: Control vs LPS (Microglia only)
################################################################################

cat(strrep("=", 80), "\n")
cat("ANALYSIS 2: CONTROL vs LPS (Microglia only)\n")
cat("="," repetition of 80 =s\n\n")

# Filter to microglia only
micro_animal <- animal_data %>% filter(Compartment == "Microglia")
micro_profile <- profile_data %>% filter(Compartment == "Microglia")

# Function for two-sample analysis with permutation test
two_sample_analysis <- function(data, antibody_type) {
  ab_data <- data %>% filter(Antibody == antibody_type)
  
  ctrl <- ab_data %>% filter(Treatment == "Control") %>% pull(Mean_Density)
  lps <- ab_data %>% filter(Treatment == "LPS") %>% pull(Mean_Density)
  
  # Welch's t-test
  t_result <- t.test(ctrl, lps, var.equal = FALSE)
  
  # Mann-Whitney U test
  mw_result <- wilcox.test(ctrl, lps, exact = FALSE)
  
  # Exact permutation test (all 15 combinations for n=4 vs n=2)
  combined <- c(ctrl, lps)
  obs_diff <- mean(ctrl) - mean(lps)
  n_perm <- choose(6, 2)
  count_extreme <- 0
  
  for (idx in combn(6, 2, simplify = FALSE)) {
    grp2 <- combined[idx]
    grp1 <- combined[-idx]
    if (abs(mean(grp1) - mean(grp2)) >= abs(obs_diff)) {
      count_extreme <- count_extreme + 1
    }
  }
  perm_p <- count_extreme / n_perm
  
  # Hedges' g
  n1 <- length(ctrl)
  n2 <- length(lps)
  pooled_sd <- sqrt(((n1-1)*var(ctrl) + (n2-1)*var(lps)) / (n1+n2-2))
  d <- (mean(ctrl) - mean(lps)) / pooled_sd
  g <- d * (1 - 3/(4*(n1+n2)-9))  # Hedges correction
  
  # Bootstrap 95% CI
  set.seed(42)
  boot_diffs <- replicate(10000, {
    b1 <- sample(ctrl, replace = TRUE)
    b2 <- sample(lps, replace = TRUE)
    mean(b1) - mean(b2)
  })
  boot_ci <- quantile(boot_diffs, c(0.025, 0.975))
  
  # Bayesian probability (using t-distribution posterior)
  se_diff <- pooled_sd * sqrt(1/n1 + 1/n2)
  posterior <- obs_diff + se_diff * rt(100000, df = n1 + n2 - 2)
  prob_ctrl_greater <- mean(posterior > 0)
  
  data.frame(
    Antibody = antibody_type,
    Control_mean = mean(ctrl),
    Control_SD = sd(ctrl),
    Control_n = n1,
    LPS_mean = mean(lps),
    LPS_SD = sd(lps),
    LPS_n = n2,
    Difference = mean(ctrl) - mean(lps),
    Pct_change = ((mean(lps) - mean(ctrl)) / mean(ctrl)) * 100,
    Welch_t_statistic = t_result$statistic,
    Welch_t_pvalue = t_result$p.value,
    Welch_t_CI_lower = t_result$conf.int[1],
    Welch_t_CI_upper = t_result$conf.int[2],
    MannWhitney_pvalue = mw_result$p.value,
    Permutation_pvalue = perm_p,
    Hedges_g = g,
    Bootstrap_CI_lower = boot_ci[1],
    Bootstrap_CI_upper = boot_ci[2],
    Bayesian_prob_ctrl_greater = prob_ctrl_greater
  )
}

# Run analyses
analysis2_tests <- rbind(
  two_sample_analysis(micro_animal, "GABA"),
  two_sample_analysis(micro_animal, "GLUT")
)

# Mixed-effects models for Control vs LPS
analysis2_mixed <- data.frame()

for (ab in c("GABA", "GLUT")) {
  ab_data <- micro_profile %>% filter(Antibody == ab)
  ab_data$Treatment <- factor(ab_data$Treatment, levels = c("Control", "LPS"))
  
  # Fit mixed model
  model <- lmer(Density_per_um2 ~ Treatment + (1|Animal), data = ab_data)
  model_summary <- summary(model)
  coefs <- as.data.frame(coef(model_summary))
  
  ci <- confint(model, parm = "beta_", method = "Wald")
  
  param_name <- "TreatmentLPS"
  analysis2_mixed <- rbind(analysis2_mixed, data.frame(
    Antibody = ab,
    Estimate = coefs[param_name, "Estimate"],
    SE = coefs[param_name, "Std. Error"],
    df = coefs[param_name, "df"],
    t_value = coefs[param_name, "t value"],
    p_value = coefs[param_name, "Pr(>|t|)"],
    CI_lower = ci[param_name, 1],
    CI_upper = ci[param_name, 2],
    N_profiles = nrow(ab_data),
    N_animals = length(unique(ab_data$Animal))
  ))
}

# Save Analysis 2 results
write.csv(analysis2_tests, file.path(results_path, "analysis2_statistical_tests.csv"), row.names = FALSE)
write.csv(analysis2_mixed, file.path(results_path, "analysis2_mixed_model.csv"), row.names = FALSE)

cat("Analysis 2 complete. Results saved.\n\n")

################################################################################
# DESCRIPTIVE STATISTICS SUMMARY
################################################################################

# Create comprehensive summary tables
descriptive_profile <- profile_data %>%
  group_by(Antibody, Treatment, Compartment) %>%
  summarise(
    N_profiles = n(),
    Mean_density = mean(Density_per_um2),
    SD_density = sd(Density_per_um2),
    SEM_density = sd(Density_per_um2) / sqrt(n()),
    Median_density = median(Density_per_um2),
    Min_density = min(Density_per_um2),
    Max_density = max(Density_per_um2),
    Total_particles = sum(N_particles_total),
    .groups = "drop"
  )

descriptive_animal <- animal_data %>%
  group_by(Antibody, Treatment, Compartment) %>%
  summarise(
    N_animals = n(),
    Mean_of_means = mean(Mean_Density),
    SD_of_means = sd(Mean_Density),
    SEM_of_means = sd(Mean_Density) / sqrt(n()),
    .groups = "drop"
  )

write.csv(descriptive_profile, file.path(results_path, "descriptive_stats_profile_level.csv"), row.names = FALSE)
write.csv(descriptive_animal, file.path(results_path, "descriptive_stats_animal_level.csv"), row.names = FALSE)

cat("Descriptive statistics saved.\n\n")

################################################################################
# PUBLICATION-READY FIGURES
################################################################################

cat("Creating publication-ready figures...\n")

# Set Nature/Science theme
theme_publication <- theme_classic(base_size = 10, base_family = "Helvetica") +
  theme(
    axis.line = element_line(color = "black", linewidth = 0.5),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    axis.text = element_text(color = "black", size = 9),
    axis.title = element_text(color = "black", size = 10, face = "bold"),
    plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
    legend.position = "none",
    panel.grid = element_blank(),
    plot.margin = margin(10, 10, 10, 10)
  )

# Color palettes
colors_compartment <- c("GLUT_terminals" = "#2E86AB", "Microglia" = "#A23B72", 
                        "GABA_terminals" = "#28965A", "Spine" = "#F18F01")
colors_treatment <- c("Control" = "#2E86AB", "LPS" = "#E94F37")

################################################################################
# FIGURE 1: Glutamate labeling across compartments (Control)
################################################################################

fig1_data <- ctrl_animal %>% 
  filter(Antibody == "GLUT") %>%
  mutate(Compartment = factor(Compartment, 
                              levels = c("GLUT_terminals", "Microglia", "GABA_terminals", "Spine")))

fig1_summary <- fig1_data %>%
  group_by(Compartment) %>%
  summarise(Mean = mean(Mean_Density), SEM = sd(Mean_Density)/sqrt(n()), .groups = "drop")

fig1 <- ggplot() +
  # Individual data points
  geom_point(data = fig1_data, 
             aes(x = Compartment, y = Mean_Density, fill = Compartment),
             shape = 21, size = 3.5, stroke = 0.5, color = "black", alpha = 0.8,
             position = position_jitter(width = 0.1, seed = 42)) +
  # Mean bars
  geom_crossbar(data = fig1_summary,
                aes(x = Compartment, y = Mean, ymin = Mean, ymax = Mean),
                width = 0.5, linewidth = 0.7, color = "black") +
  # Error bars (SEM)
  geom_errorbar(data = fig1_summary,
                aes(x = Compartment, ymin = Mean - SEM, ymax = Mean + SEM),
                width = 0.2, linewidth = 0.5, color = "black") +
  # Significance annotations
  annotate("segment", x = 1, xend = 2, y = 105, yend = 105, linewidth = 0.4) +
  annotate("text", x = 1.5, y = 110, label = "**", size = 4) +
  annotate("segment", x = 2, xend = 4, y = 55, yend = 55, linewidth = 0.4) +
  annotate("text", x = 3, y = 60, label = "**", size = 4) +
  scale_fill_manual(values = colors_compartment) +
  scale_x_discrete(labels = c("GLUT\nterminals", "Microglia", "GABA\nterminals", "Spine")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)), limits = c(0, NA)) +
  labs(
    title = "Glutamate immunolabeling",
    x = NULL,
    y = expression(bold("Gold particle density (particles/"*mu*"m"^2*")"))
  ) +
  theme_publication

ggsave(file.path(figures_path, "Figure1_glutamate_compartments.pdf"), fig1, width = 4, height = 4.5, dpi = 300)
ggsave(file.path(figures_path, "Figure1_glutamate_compartments.png"), fig1, width = 4, height = 4.5, dpi = 300)

################################################################################
# FIGURE 2: GABA labeling across compartments (Control)
################################################################################

fig2_data <- ctrl_animal %>% 
  filter(Antibody == "GABA") %>%
  mutate(Compartment = factor(Compartment, 
                              levels = c("GABA_terminals", "Microglia", "GLUT_terminals", "Spine")))

fig2_summary <- fig2_data %>%
  group_by(Compartment) %>%
  summarise(Mean = mean(Mean_Density), SEM = sd(Mean_Density)/sqrt(n()), .groups = "drop")

fig2 <- ggplot() +
  geom_point(data = fig2_data, 
             aes(x = Compartment, y = Mean_Density, fill = Compartment),
             shape = 21, size = 3.5, stroke = 0.5, color = "black", alpha = 0.8,
             position = position_jitter(width = 0.1, seed = 42)) +
  geom_crossbar(data = fig2_summary,
                aes(x = Compartment, y = Mean, ymin = Mean, ymax = Mean),
                width = 0.5, linewidth = 0.7, color = "black") +
  geom_errorbar(data = fig2_summary,
                aes(x = Compartment, ymin = Mean - SEM, ymax = Mean + SEM),
                width = 0.2, linewidth = 0.5, color = "black") +
  # Significance annotations
  annotate("segment", x = 1, xend = 2, y = 340, yend = 340, linewidth = 0.4) +
  annotate("text", x = 1.5, y = 355, label = "***", size = 4) +
  annotate("segment", x = 2, xend = 3, y = 35, yend = 35, linewidth = 0.4) +
  annotate("text", x = 2.5, y = 42, label = "***", size = 4) +
  annotate("segment", x = 2, xend = 4, y = 48, yend = 48, linewidth = 0.4) +
  annotate("text", x = 3, y = 55, label = "***", size = 4) +
  scale_fill_manual(values = colors_compartment) +
  scale_x_discrete(labels = c("GABA\nterminals", "Microglia", "GLUT\nterminals", "Spine")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)), limits = c(0, NA)) +
  labs(
    title = "GABA immunolabeling",
    x = NULL,
    y = expression(bold("Gold particle density (particles/"*mu*"m"^2*")"))
  ) +
  theme_publication

ggsave(file.path(figures_path, "Figure2_GABA_compartments.pdf"), fig2, width = 4, height = 4.5, dpi = 300)
ggsave(file.path(figures_path, "Figure2_GABA_compartments.png"), fig2, width = 4, height = 4.5, dpi = 300)

################################################################################
# FIGURE 3: Control vs LPS comparison (Microglia)
################################################################################

fig3_data <- micro_animal %>%
  mutate(Treatment = factor(Treatment, levels = c("Control", "LPS")))

fig3_summary <- fig3_data %>%
  group_by(Antibody, Treatment) %>%
  summarise(Mean = mean(Mean_Density), SEM = sd(Mean_Density)/sqrt(n()), .groups = "drop")

# Panel A: GABA
fig3a_data <- fig3_data %>% filter(Antibody == "GABA")
fig3a_summary <- fig3_summary %>% filter(Antibody == "GABA")

fig3a <- ggplot() +
  geom_point(data = fig3a_data, 
             aes(x = Treatment, y = Mean_Density, fill = Treatment),
             shape = 21, size = 4, stroke = 0.5, color = "black", alpha = 0.8,
             position = position_jitter(width = 0.08, seed = 42)) +
  geom_crossbar(data = fig3a_summary,
                aes(x = Treatment, y = Mean, ymin = Mean, ymax = Mean),
                width = 0.4, linewidth = 0.7, color = "black") +
  geom_errorbar(data = fig3a_summary,
                aes(x = Treatment, ymin = Mean - SEM, ymax = Mean + SEM),
                width = 0.15, linewidth = 0.5, color = "black") +
  annotate("segment", x = 1, xend = 2, y = 28, yend = 28, linewidth = 0.4) +
  annotate("text", x = 1.5, y = 30, label = "p = 0.067", size = 3) +
  scale_fill_manual(values = colors_treatment) +
  scale_x_discrete(labels = c("Control\n(n=4)", "LPS\n(n=2)")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)), limits = c(0, 35)) +
  labs(
    title = "GABA in microglia",
    x = NULL,
    y = expression(bold("Gold particle density (particles/"*mu*"m"^2*")"))
  ) +
  theme_publication

# Panel B: Glutamate
fig3b_data <- fig3_data %>% filter(Antibody == "GLUT")
fig3b_summary <- fig3_summary %>% filter(Antibody == "GLUT")

fig3b <- ggplot() +
  geom_point(data = fig3b_data, 
             aes(x = Treatment, y = Mean_Density, fill = Treatment),
             shape = 21, size = 4, stroke = 0.5, color = "black", alpha = 0.8,
             position = position_jitter(width = 0.08, seed = 42)) +
  geom_crossbar(data = fig3b_summary,
                aes(x = Treatment, y = Mean, ymin = Mean, ymax = Mean),
                width = 0.4, linewidth = 0.7, color = "black") +
  geom_errorbar(data = fig3b_summary,
                aes(x = Treatment, ymin = Mean - SEM, ymax = Mean + SEM),
                width = 0.15, linewidth = 0.5, color = "black") +
  annotate("segment", x = 1, xend = 2, y = 52, yend = 52, linewidth = 0.4) +
  annotate("text", x = 1.5, y = 55, label = "n.s.", size = 3) +
  scale_fill_manual(values = colors_treatment) +
  scale_x_discrete(labels = c("Control\n(n=4)", "LPS\n(n=2)")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)), limits = c(0, 60)) +
  labs(
    title = "Glutamate in microglia",
    x = NULL,
    y = expression(bold("Gold particle density (particles/"*mu*"m"^2*")"))
  ) +
  theme_publication

# Combine panels
fig3_combined <- plot_grid(fig3a, fig3b, labels = c("A", "B"), label_size = 12, ncol = 2)

ggsave(file.path(figures_path, "Figure3_Control_vs_LPS.pdf"), fig3_combined, width = 7, height = 4, dpi = 300)
ggsave(file.path(figures_path, "Figure3_Control_vs_LPS.png"), fig3_combined, width = 7, height = 4, dpi = 300)

################################################################################
# FIGURE 4: Combined overview figure
################################################################################

# Create a 2x2 combined figure
fig_combined <- plot_grid(
  fig1, fig2, fig3a, fig3b,
  labels = c("A", "B", "C", "D"),
  label_size = 12,
  ncol = 2,
  nrow = 2
)

ggsave(file.path(figures_path, "Figure_combined_overview.pdf"), fig_combined, width = 8, height = 8, dpi = 300)
ggsave(file.path(figures_path, "Figure_combined_overview.png"), fig_combined, width = 8, height = 8, dpi = 300)

cat("\nAll figures saved!\n")
cat("Output files:\n")
cat("  - Figure1_glutamate_compartments.pdf/png\n")
cat("  - Figure2_GABA_compartments.pdf/png\n")
cat("  - Figure3_Control_vs_LPS.pdf/png\n")
cat("  - Figure_combined_overview.pdf/png\n")

################################################################################
# SAVE SESSION INFO
################################################################################

sink(file.path(results_path, "R_session_info.txt"))
cat("Analysis completed:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
sessionInfo()
sink()

cat("\nAnalysis complete!\n")
cat("\nAll files saved to:\n")
cat("  Results:", results_path, "\n")
cat("  Figures:", figures_path, "\n")

