# This script performs all the statistical testing for the analysis in
# "Anterior cingulate neurons combine outcome monitoring of past decisions
# with ongoing movement signals", by Oesch et al.
#
# The analyses here assume that all the results data frames are saved as .csv
# files into the same file directory (base_dir, to be set by the user below).
# To export the results from statistical analyses use the following:
# write.csv(summary(fit)$coefficients, "path_to_my_file.csv") for lme results, and
# write.csv(summary(emm, by = NULL, adjust = "sidak")$contrasts,"path_to_my_file.csv")
# for the contrasts on the computed marginal means with p-values adjusted for 
# multiple comparison using sidak's method.
#
# LO, February 2026
#-------------------------------------------------------------------------------

#Importing
library(lmerTest)
library(ggplot2)
library(emmeans)

#Define the base directory for all the analysis files
base_dir <- "/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/data_tables"

# --------------Figure 1F----------------------------------------
# Comparing the magnitudes of the model weights to the shuffled control
data <- read.csv(file.path(base_dir, "Fig1F_logreg_weights.csv"), header=TRUE, sep=",")
data$is_shuffle <- relevel(factor(data$is_shuffle), ref="shuffle")
fit <- lmerTest::lmer(abs_logreg_weights ~ regressor * is_shuffle +  (1 |subject), data = data) 
summary(fit) #
emm = emmeans(fit, specs = pairwise ~ is_shuffle | regressor, pbkrtest.limit = 4272)
summary(emm, by=NULL, adjust = "sidak")

# -------------Figure 1H-----------------------------------------
# Fitting linear relationship between the difference between history strength
# and stimulus weight, and performance 
data <- read.csv(file.path(base_dir, "Fig1H_hist_delta_vs_performance.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(performance ~ hist_stim_coef_delta + (1 |subject), data = data, REML=FALSE) 
summary(fit)

# -----------Figure 2F----------------------------------------------
# Test for decoding accuracy difference between data and shuffle
data <- read.csv(file.path(base_dir, "Fig2F_decoding_accuracy_decoder_vs_shuffle.csv"), header=TRUE, sep=",")
data$condition <- relevel(factor(data$condition), ref="shuffle") #Set shuffle to be the reference class
fit <- lmerTest::lmer(accuracy ~ condition * time + (1  |subject/session), data = data, REML=FALSE) 
summary(fit)

# -----------Figure 2G---------------------------------------------
# Comparing the decoding accuracy between the different trial history contexts
data <- read.csv(file.path(base_dir, "Fig2G_class_wise_decoding_accuracy.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(accuracy ~ history_context * time + (1 |subject/session), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ history_context | time, pbkrtest.limit = 12236)
summary(emm, by = NULL, adjust = "sidak")

# ------------Figure 2J-----------------------------------------------
# Compare cross-decoding accuracy with shuffled control
data <- read.csv(file.path(base_dir, "Fig2J_phase_wise_cross_decoding.csv"), header=TRUE, sep=",")
data$condition <- relevel(factor(data$condition), ref="shuffle")
data$seed <- relevel(factor(data$seed), ref="Early ITI") #Seed here refers to the phase the decoder was trained in
data$phase <- relevel(factor(data$phase), ref="Early ITI") #Phase is the testing phase
fit <- lmerTest::lmer(accuracy ~ seed*phase*condition + (1  |subject/session), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ condition | seed*phase)
summary(emm, by = NULL, adjust = "sidak")

# -----------Figure 4D------------------------------------------------
#Compare the dimensionality of trial history, chest and video encoding
data <- read.csv(file.path(base_dir, "Fig4D_encoding_weight_dimensionality.csv"), header=TRUE, sep=",")
data$regressor <- relevel(factor(data$regressor), ref="previous_choice_outcome_combination")
fit <- lmerTest::lmer(dimensionality ~ regressor +  (1 |subject/session), data = data, REML = FALSE) #Here regressor refers to chest, trial history, etc
summary(fit)
emm = emmeans(fit, specs = pairwise ~ regressor )
summary(emm, by = NULL, adjust = "sidak")

# -----------Figure 4G-----------------------------------------------
#Compare the within- and across subject procrustes distances for trial history, chest and video
data <- read.csv(file.path(base_dir, "Fig4G_procrustes_distance.csv"), header=TRUE, sep=",")
data$regressor <- relevel(factor(data$regressor), ref="previous_choice_outcome_combination")
data$condition <- relevel(factor(data$condition), ref="within")
fit <- lmerTest::lmer(procrustes_distance ~ regressor * condition + (1 |subject), data = data, REML = FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ regressor *condition)
summary(emm, by = NULL, adjust = "sidak")

# ------------Supplementary Figure 1A----------------------------------
# Perceptual bias
data <- read.csv(file.path(base_dir, "Supplementary_Fig1A_perceptual_bias.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(param_estimate ~ trial_history + (1 |subject), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ trial_history)
summary(emm, by = NULL, adjust = "sidak")

# ------------Supplementary Figure 1B----------------------------------
# Sensitivity
data <- read.csv(file.path(base_dir, "Supplementary_Fig1B_sensitivity.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(param_estimate ~ trial_history + (1 |subject), data = data, REML=FALSE) 
summary(fit)

# ------------Supplementary Figure 1C----------------------------------
# Sensitivity
data <- read.csv(file.path(base_dir, "Supplementary_Fig1C_upper_lapse_rate.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(param_estimate ~ trial_history + (1 |subject), data = data, REML=FALSE) 
summary(fit)

# ------------Supplementary Figure 1D----------------------------------
# Sensitivity
data <- read.csv(file.path(base_dir, "Supplementary_Fig1D_lower_lapse_rate.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(param_estimate ~ trial_history + (1 |subject), data = data, REML=FALSE) 
summary(fit)

# ----------Supplementary Figure 1E--------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig1E_weigth_variance_no_shuffle.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(weight_variance ~ regressor + (1 |subject), data = data, REML=FALSE) 
summary(fit)

# ----------Supplementary Figure 1G------------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig1G_main_effects_logreg_weights.csv"), header=TRUE, sep=",")
data$is_shuffle <- relevel(factor(data$is_shuffle), ref="shuffle")
data$regressor <- relevel(factor(data$regressor), ref="intercept")
fit <- lmerTest::lmer(abs_logreg_weights ~ regressor * is_shuffle +  (1 |subject), data = data) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ regressor | is_shuffle, pbkrtest.limit = 5632)
summary(emm, by = NULL, adjust = "sidak")

#------------Supplementary Figure 2B----------------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig2_decoding_accuracy_different_decoders.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(accuracy ~ decoded_variable * is_history + (1 |subject/session), data = data, REML = FALSE)
summary(fit)

#------------Supplementary Figure 2C----------------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig2C_stim_phase_decoding_accuracy_by_ITI_duration.csv"), header=TRUE, sep=",")
data$condition <- relevel(factor(data$condition), ref = "Shuffle")  
fit <- lmerTest::lmer(decoding_accuracy ~ condition * ITI_bin + (1 | subject/session), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ condition | ITI_bin)
summary(emm, by = NULL, adjust = "sidak")

#------------Supplementary Figure 2D----------------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig2D_Stim_phase_decoding_accuracy_by_ITI_duration_hist_context.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(decoding_accuracy ~ hist_context * ITI_bin + (1 | subject/session), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ hist_context | ITI_bin)
summary(emm, by = NULL, adjust = "sidak")

#------------Supplementary Figure 2E----------------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig2E_history_decoding_t-2_overall.csv"), , header=TRUE, sep=",")
data$condition <- relevel(factor(data$condition), ref = "Shuffle")  
fit <- lmerTest::lmer(decoding_accuracy ~ condition * time + (1 | subject/session), data = data, REML=FALSE) 
summary(fit)

#------------Supplementary Figure 2F----------------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig2F_history_decoding_t-2_by_context.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(decoding_accuracy ~ hist_context * time + (1 | subject/session), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ hist_context | time, pbkrtest.limit = 11172)
summary(emm, by = NULL, adjust = "sidak")

# -----------Supplementary Figure 5B and E---------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig5.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(decoding_accuracy ~ lens_AP * lens_DV + sex + (1 | subject), data = data, REML=FALSE) 
summary(fit)

# ---------Supplementary Figure 5C and F----------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig5.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(cvR ~ lens_AP * lens_DV + sex  + (1  |subject), data = data, REML=FALSE) 
summary(fit)

# --------Supplementary Figure 5D and G-----------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig5.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(dR ~ lens_AP * lens_DV + sex  +  (1 |subject), data = data, REML=FALSE) 
summary(fit)

#---------Reviewer Figure 2A----------------------------------------------------
# Test for decoding accuracy difference between data and shuffle for all sessions with no overlap between ITI phases
data <- read.csv(file.path(base_dir, "Reviewer_Figure2a_decoding_accuracy_decoder_vs_shuffle_no_overlap.csv"), header=TRUE, sep=",")
data$condition <- relevel(factor(data$condition), ref="shuffle") #Set shuffle to be the reference class
fit <- lmerTest::lmer(accuracy ~ condition * time + (1  |subject/session), data = data, REML=FALSE) 
summary(fit)

# -----------Reviewer Figure 2B---------------------------------------------
# Comparing the decoding accuracy between the different trial history contexts for sessions with no overlap between ITI phases
data <- read.csv(file.path(base_dir, "Reviewer_figure2b_class_wise_decoding_accuracy_no_overlap.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(accuracy ~ history_context * time + (1 |subject/session), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ history_context | time, pbkrtest.limit = 12236)
summary(emm, by = NULL, adjust = "sidak")

# ------------Reviewer Figure 2E-----------------------------------------------
# Compare cross-decoding accuracy with shuffled control for sessions with no overlap for the two ITI phases
data <- read.csv(file.path(base_dir, "Reviewer_Figure2e_phase_wise_cross_decoding_no_overlap.csv"), header=TRUE, sep=",")
data$condition <- relevel(factor(data$condition), ref="shuffle")
data$seed <- relevel(factor(data$seed), ref="Early ITI") #Seed here refers to the phase the decoder was trained in
data$phase <- relevel(factor(data$phase), ref="Early ITI") #Phase is the testing phase
fit <- lmerTest::lmer(accuracy ~ seed*phase*condition + (1  |subject/session), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ condition | seed*phase)
summary(emm, by = NULL, adjust = "sidak")

# -------------Reviewer Figure 3A-----------------------------------------
# Fitting linear relationship between the difference between history strength
# and stimulus weight, and performance excluding one animal (LY008) with sessions
# showing a large difference between history strength and stimulus weight.
data <- read.csv(file.path(base_dir, "Reviewer_figure3a_hist_delta_vs_performance.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(performance ~ hist_stim_coef_delta + (1 |subject), data = data, REML=FALSE) 
summary(fit)

