library(lmerTest)
library(dplyr)
library(performance)

outcome_variable <- "sleepiness"
data <- read.csv(paste0("processed_data/42-", outcome_variable, "-multimodal.csv"))

data[[outcome_variable]] <- as.numeric(data[[outcome_variable]])
output_file <- paste0("processed_data/42-", outcome_variable, "-multimodal-output.csv")

exclude_cols <- c("participant", outcome_variable)
predictors <- setdiff(names(data), exclude_cols)

# 建议：null model 只拟合一次，避免循环里重复计算
null_model <- lmer(as.formula(paste0(outcome_variable, " ~ (1 | participant)")),
                   data = data, REML = FALSE)

results <- data.frame(
  Predictor = character(),
  Term = character(),          # 新增：具体系数名（数值变量=Predictor；分类变量=Predictor+水平）
  Estimate = numeric(),
  CI_Lower = numeric(),
  CI_Upper = numeric(),
  P_value = numeric(),
  Marginal_R2 = numeric(),
  Conditional_R2 = numeric(),
  AIC = numeric(),
  BIC = numeric(),
  NegLogLik = numeric(),
  ANOVA_P_value = numeric(),
  stringsAsFactors = FALSE
)

for (predictor in predictors) {
  formula <- as.formula(paste0(outcome_variable, " ~ ", predictor, " + (1 | participant)"))
  
  model <- lmer(formula, data = data, REML = FALSE)
  
  # 新增：marginal R^2（固定效应）
  r2_vals <- performance::r2(model)
  marginal_r2 <- as.numeric(r2_vals$R2_marginal)
  conditional_r2 <- as.numeric(r2_vals$R2_conditional)
  model_summary <- summary(model)
  coef_tab <- model_summary$coefficients
  
  # 找到该 predictor 对应的所有系数行名（数值：通常就是 predictor；分类：predictorXXX 多行）
  term_names <- grep(paste0("^", predictor), rownames(coef_tab), value = TRUE)
  
  # 有些情况下（比如完全共线/奇异拟合）可能找不到，做个保护
  if (length(term_names) == 0) next
  
  # 你原来的 LRT（整体效应）照旧：对“整个 predictor（含所有水平）”做比较
  anova_result <- anova(null_model, model)
  anova_p_value <- anova_result[2, "Pr(>Chisq)"]
  
  aic <- AIC(model)
  bic <- BIC(model)
  neg_log_lik <- as.numeric(-logLik(model))
  
  # 用 Wald CI 更快更稳（profile CI 常常很慢/失败）
  ci_mat <- suppressMessages(confint(model, parm = term_names, method = "Wald"))
  
  for (term in term_names) {
    results <- rbind(results, data.frame(
      Predictor = predictor,
      Term = term,
      Estimate = coef_tab[term, "Estimate"],
      CI_Lower = ci_mat[term, 1],
      CI_Upper = ci_mat[term, 2],
      P_value = coef_tab[term, "Pr(>|t|)"],
      Marginal_R2 = marginal_r2,
      Conditional_R2 = conditional_r2,
      AIC = aic,
      BIC = bic,
      NegLogLik = neg_log_lik,
      ANOVA_P_value = anova_p_value,
      stringsAsFactors = FALSE
    ))
  }
}

# write.csv(results, file = output_file, row.names = FALSE)
cat("Results saved to:", output_file, "\n")