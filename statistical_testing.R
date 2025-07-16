
#Importing
library(lmerTest)
library(ggplot2)
library(emmeans)

#Define the base directory for all the analysis files
base_dir <- "/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/data_tables"

# --------------Figure 1E----------------------------------------
# Comparing the magnitudes of the model weights to the shuffled control
data <- read.csv(file.path(base_dir, "Fig1E_logreg_weights.csv"), header=TRUE, sep=",")
data$is_shuffle <- relevel(factor(data$is_shuffle), ref="shuffle")
fit <- lmerTest::lmer(abs_logreg_weights ~ regressor * is_shuffle +  (1 |subject), data = data) 
summary(fit) #
emm = emmeans(fit, specs = pairwise ~ is_shuffle | regressor, pbkrtest.limit = 4272)
summary(emm)

# -------------Figure 1G-----------------------------------------
# Fitting linear relationship between the difference between history strength
# and stimulus weight, and performance 
data <- read.csv(file.path(base_dir, "Fig1G_hist_delta_vs_performance.csv"), header=TRUE, sep=",")
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
summary(emm)

# ------------Figure 2J-----------------------------------------------
# Compare cross-decoding accuracy with shuffled control
data <- read.csv(file.path(base_dir, "Fig2J_phase_wise_cross_decoding.csv"), header=TRUE, sep=",")
data$condition <- relevel(factor(data$condition), ref="shuffle")
data$seed <- relevel(factor(data$seed), ref="Early ITI") #Seed here refers to the phase the decoder was trained in
data$phase <- relevel(factor(data$phase), ref="Early ITI") #Phase is the testing phase
fit <- lmerTest::lmer(accuracy ~ seed*phase*condition + (1  |subject/session), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ condition | seed*phase)
summary(emm)

# -----------Figure 4D------------------------------------------------
#Compare the dimensionality of trial history, chest and video encoding
data <- read.csv(file.path(base_dir, "Fig4D_encoding_weight_dimensionality.csv"), header=TRUE, sep=",")
data$regressor <- relevel(factor(data$regressor), ref="previous_choice_outcome_combination")
fit <- lmerTest::lmer(dimensionality ~ regressor +  (1 |subject/session), data = data, REML = FALSE) #Here regressor refers to chest, trial history, etc
summary(fit)
emm = emmeans(fit, specs = pairwise ~ regressor )
summary(emm)

# -----------Figure 4G-----------------------------------------------
#Compare the within- and across subject procrustes distances for trial history, chest and video
data <- read.csv(file.path(base_dir, "Fig4G_procrustes_distance.csv"), header=TRUE, sep=",")
data$regressor <- relevel(factor(data$regressor), ref="previous_choice_outcome_combination")
data$condition <- relevel(factor(data$condition), ref="within")
fit <- lmerTest::lmer(procrustes_distance ~ regressor * condition + (1 |subject), data = data, REML = FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ regressor *condition)
summary(emm)

# ------------Supplementary Figure 1A----------------------------------
# Perceptual bias
data <- read.csv(file.path(base_dir, "Supplementary_Fig1A_perceptual_bias.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(param_estimate ~ trial_history + (1 |subject), data = data, REML=FALSE) 
summary(fit)
emm = emmeans(fit, specs = pairwise ~ trial_history)
summary(emm)

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
summary(emm)

#------------Supplementary Figure 2B----------------------------------------------
data <- read.csv(file.path(base_dir, "Supplementary_Fig2_decoding_accuracy_different_decoders.csv"), header=TRUE, sep=",")
fit <- lmerTest::lmer(accuracy ~ decoded_variable * is_history + (1 |subject/session), data = data, REML = FALSE)
summary(fit)

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
