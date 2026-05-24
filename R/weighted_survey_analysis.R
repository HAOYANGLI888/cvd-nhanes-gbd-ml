suppressPackageStartupMessages({
  library(optparse)
  library(survey)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(splines)
})

option_list <- list(
  make_option("--project-root", dest = "project_root", type = "character", default = "."),
  make_option("--input", dest = "input", type = "character", default = NA, help = "Default: data/processed/nhanes_analysis.csv"),
  make_option("--ranking", dest = "ranking", type = "character", default = NA, help = "Default: outputs/tables/table3_feature_ranking.csv"),
  make_option("--weight", dest = "weight", type = "character", default = "survey_weight"),
  make_option("--top-n", dest = "top_n", type = "integer", default = 10),
  make_option("--rcs-indicator", dest = "rcs_indicator", type = "character", default = NA),
  make_option("--log-file", dest = "log_file", type = "character", default = NA)
)

opt <- parse_args(OptionParser(option_list = option_list))

project_root <- normalizePath(opt$project_root, winslash = "/", mustWork = FALSE)
input_path <- ifelse(is.na(opt$input), file.path(project_root, "data/processed/nhanes_analysis.csv"), opt$input)
ranking_path <- ifelse(is.na(opt$ranking), file.path(project_root, "outputs/tables/table3_feature_ranking.csv"), opt$ranking)
tables_dir <- file.path(project_root, "outputs/tables")
figures_dir <- file.path(project_root, "outputs/figures")
logs_dir <- file.path(project_root, "outputs/logs")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- ifelse(is.na(opt$log_file), file.path(logs_dir, "weighted_survey_analysis.log"), opt$log_file)

log_msg <- function(...) {
  msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste(..., collapse = " "))
  cat(msg, "\n")
  cat(msg, "\n", file = log_file, append = TRUE)
}

present <- function(cols, data) cols[cols %in% names(data)]

weighted_quantiles <- function(design, var, probs = c(0.25, 0.5, 0.75)) {
  f <- as.formula(paste0("~", var))
  q <- tryCatch(
    as.numeric(svyquantile(f, design, quantiles = probs, na.rm = TRUE, ci = FALSE)),
    error = function(e) as.numeric(stats::quantile(design$variables[[var]], probs = probs, na.rm = TRUE))
  )
  unique(q)
}

mode_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

safe_formula <- function(outcome, terms) {
  rhs <- paste(terms, collapse = " + ")
  as.formula(paste(outcome, "~", rhs))
}

tidy_or <- function(fit, term, label, model_name, indicator, contrast = "continuous") {
  sm <- summary(fit)$coefficients
  if (!(term %in% rownames(sm))) return(NULL)
  beta <- sm[term, "Estimate"]
  se <- sm[term, "Std. Error"]
  p <- sm[term, ncol(sm)]
  data.frame(
    indicator = indicator,
    model = model_name,
    contrast = contrast,
    term = label,
    beta = beta,
    se = se,
    OR = exp(beta),
    CI_low = exp(beta - 1.96 * se),
    CI_high = exp(beta + 1.96 * se),
    p_value = p,
    stringsAsFactors = FALSE
  )
}

tidy_factor_or <- function(fit, indicator, model_name, contrast = "level vs reference") {
  sm <- summary(fit)$coefficients
  terms <- rownames(sm)
  indicator_terms <- terms[startsWith(terms, indicator) & terms != indicator]
  if (length(indicator_terms) == 0) return(NULL)

  bind_rows(lapply(indicator_terms, function(term) {
    beta <- sm[term, "Estimate"]
    se <- sm[term, "Std. Error"]
    p <- sm[term, ncol(sm)]
    data.frame(
      indicator = indicator,
      model = model_name,
      contrast = contrast,
      term = sub(paste0("^", indicator), "", term),
      beta = beta,
      se = se,
      OR = exp(beta),
      CI_low = exp(beta - 1.96 * se),
      CI_high = exp(beta + 1.96 * se),
      p_value = p,
      stringsAsFactors = FALSE
    )
  }))
}

is_continuous_indicator <- function(data, var) {
  is.numeric(data[[var]]) && length(unique(na.omit(data[[var]]))) > 3
}

make_design <- function(data, weight_col) {
  options(survey.lonely.psu = "adjust")
  svydesign(
    ids = as.formula("~psu"),
    strata = as.formula("~strata"),
    weights = as.formula(paste0("~", weight_col)),
    nest = TRUE,
    data = data
  )
}

make_table1 <- function(design, data) {
  continuous <- present(c(
    "age", "pir", "bmi", "waist", "mean_sbp", "mean_dbp", "fasting_glucose", "hba1c",
    "total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol",
    "tyg_index", "tyg_bmi", "tyg_wc", "aip", "nlr", "plr", "sii", "siri",
    "egfr", "uacr", "uric_acid", "crp", "sleep_hours"
  ), data)
  categorical <- present(c(
    "sex", "race_ethnicity", "education", "smoking_status", "alcohol_use",
    "physical_activity", "hypertension", "diabetes"
  ), data)

  rows <- list()
  for (v in continuous) {
    group_values <- list()
    for (g in c(0, 1)) {
      dsub <- subset(design, cvd == g)
      estimate <- tryCatch(svymean(as.formula(paste0("~", v)), dsub, na.rm = TRUE), error = function(e) NULL)
      if (is.null(estimate)) {
        group_values[[as.character(g)]] <- NA
      } else {
        group_values[[as.character(g)]] <- sprintf("%.2f (SE %.2f)", coef(estimate)[1], SE(estimate)[1])
      }
    }
    p <- tryCatch(svyttest(as.formula(paste0(v, " ~ cvd")), design)$p.value, error = function(e) NA_real_)
    rows[[length(rows) + 1]] <- data.frame(
      variable = v, level = "Mean (SE)", non_cvd = group_values[["0"]],
      cvd = group_values[["1"]], p_value = p, stringsAsFactors = FALSE
    )
  }

  for (v in categorical) {
    f <- as.formula(paste0("~", v, " + cvd"))
    tab <- tryCatch(svytable(f, design), error = function(e) NULL)
    p <- tryCatch(svychisq(f, design, statistic = "F")$p.value, error = function(e) NA_real_)
    if (is.null(tab)) next
    props <- prop.table(tab, margin = 2)
    for (lvl in rownames(tab)) {
      rows[[length(rows) + 1]] <- data.frame(
        variable = v,
        level = lvl,
        non_cvd = sprintf("%.1f%%", 100 * props[lvl, "0"]),
        cvd = sprintf("%.1f%%", 100 * props[lvl, "1"]),
        p_value = p,
        stringsAsFactors = FALSE
      )
    }
  }
  bind_rows(rows)
}

read_top_indicators <- function(data, ranking_path, top_n) {
  priority <- c("tyg_index", "tyg_bmi", "tyg_wc", "aip", "sii", "nlr", "plr", "siri", "uacr", "egfr", "uric_acid", "crp")
  candidates <- character(0)
  if (file.exists(ranking_path)) {
    ranking <- suppressMessages(read_csv(ranking_path, show_col_types = FALSE))
    if ("feature" %in% names(ranking)) candidates <- ranking$feature
  }
  candidates <- unique(c(candidates, priority))
  candidates <- candidates[candidates %in% names(data)]
  head(candidates, top_n)
}

fit_indicator_models <- function(design, data, indicators) {
  base2 <- present(c("age", "sex", "race_ethnicity"), data)
  base3 <- present(c("age", "sex", "race_ethnicity", "education", "pir", "smoking_status", "alcohol_use", "physical_activity", "bmi", "hypertension", "diabetes", "total_cholesterol", "hdl_cholesterol", "triglycerides"), data)
  rows <- list()

  for (indicator in indicators) {
    for (spec in list(
      list(name = "Model 1 unadjusted", covars = character(0)),
      list(name = "Model 2 age-sex-race", covars = base2),
      list(name = "Model 3 fully adjusted", covars = base3)
    )) {
      covars <- setdiff(spec$covars, indicator)
      terms <- c(indicator, covars)
      fit <- tryCatch(svyglm(safe_formula("cvd", terms), design = design, family = quasibinomial()), error = function(e) NULL)
      if (!is.null(fit)) {
        if (is.numeric(data[[indicator]])) {
          item <- tidy_or(fit, indicator, indicator, spec$name, indicator, "per 1-unit increase")
        } else {
          item <- tidy_factor_or(fit, indicator, spec$name, "level vs reference")
        }
        if (!is.null(item)) rows[[length(rows) + 1]] <- item
      }

      if (is_continuous_indicator(data, indicator)) {
        q <- weighted_quantiles(design, indicator)
      } else {
        q <- numeric(0)
      }
      if (length(q) >= 3) {
        qvar <- paste0(indicator, "_quartile")
        trend_var <- paste0(indicator, "_quartile_median")
        dtmp <- design
        quart <- cut(dtmp$variables[[indicator]], breaks = c(-Inf, q[1:3], Inf), labels = paste0("Q", 1:4), include.lowest = TRUE)
        dtmp$variables[[qvar]] <- relevel(factor(quart), ref = "Q1")
        med <- tapply(dtmp$variables[[indicator]], quart, median, na.rm = TRUE)
        dtmp$variables[[trend_var]] <- as.numeric(med[as.character(quart)])
        fit_q <- tryCatch(svyglm(safe_formula("cvd", c(qvar, covars)), design = dtmp, family = quasibinomial()), error = function(e) NULL)
        if (!is.null(fit_q)) {
          for (lvl in paste0(qvar, "Q", 2:4)) {
            item <- tidy_or(fit_q, lvl, sub(qvar, "", lvl), spec$name, indicator, "quartile vs Q1")
            if (!is.null(item)) rows[[length(rows) + 1]] <- item
          }
        }
        fit_trend <- tryCatch(svyglm(safe_formula("cvd", c(trend_var, covars)), design = dtmp, family = quasibinomial()), error = function(e) NULL)
        if (!is.null(fit_trend)) {
          trend <- tidy_or(fit_trend, trend_var, "P for trend", spec$name, indicator, "quartile median trend")
          if (!is.null(trend)) rows[[length(rows) + 1]] <- trend
        }
      }
    }
  }
  bind_rows(rows)
}

reference_data <- function(data, indicator, covars, values) {
  ref <- data.frame(tmp_indicator = values)
  names(ref)[1] <- indicator
  for (v in covars) {
    if (v == indicator) next
    if (is.numeric(data[[v]])) {
      ref[[v]] <- median(data[[v]], na.rm = TRUE)
    } else {
      mv <- mode_value(data[[v]])
      ref[[v]] <- factor(mv, levels = levels(factor(data[[v]])))
    }
  }
  ref
}

plot_rcs <- function(design, data, indicator, figures_dir) {
  if (!is_continuous_indicator(data, indicator)) return(FALSE)
  covars <- present(c("age", "sex", "race_ethnicity", "education", "pir", "smoking_status", "alcohol_use", "physical_activity", "bmi", "hypertension", "diabetes", "total_cholesterol", "hdl_cholesterol", "triglycerides"), data)
  covars <- setdiff(covars, indicator)
  formula_text <- paste("cvd ~ ns(", indicator, ", df = 4)", ifelse(length(covars) > 0, paste("+", paste(covars, collapse = " + ")), ""))
  fit <- tryCatch(svyglm(as.formula(formula_text), design = design, family = quasibinomial()), error = function(e) NULL)
  if (is.null(fit)) return(FALSE)
  rng <- stats::quantile(data[[indicator]], probs = c(0.01, 0.99), na.rm = TRUE)
  values <- seq(rng[1], rng[2], length.out = 100)
  newdata <- reference_data(data, indicator, covars, values)
  pred <- predict(fit, newdata = newdata, type = "link", se.fit = TRUE)
  if (is.list(pred)) {
    pred_fit <- as.numeric(pred$fit)
    pred_se <- as.numeric(pred$se.fit)
  } else {
    pred_fit <- as.numeric(pred)
    pred_var <- attr(pred, "var")
    if (is.null(pred_var)) {
      pred_se <- rep(NA_real_, length(pred_fit))
    } else if (length(dim(pred_var)) == 2 && nrow(pred_var) == length(pred_fit) && ncol(pred_var) == length(pred_fit)) {
      pred_se <- sqrt(diag(pred_var))
    } else {
      pred_se <- sqrt(as.numeric(pred_var))
    }
  }
  refdata <- reference_data(data, indicator, covars, median(data[[indicator]], na.rm = TRUE))
  ref <- as.numeric(predict(fit, newdata = refdata, type = "link"))
  plot_df <- data.frame(
    indicator_value = values,
    OR = exp(pred_fit - ref),
    low = exp(pred_fit - 1.96 * pred_se - ref),
    high = exp(pred_fit + 1.96 * pred_se - ref)
  )
  p <- ggplot(plot_df, aes(x = indicator_value, y = OR)) +
    geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.18, fill = "#4E79A7") +
    geom_line(color = "#1F4E79", linewidth = 1) +
    geom_hline(yintercept = 1, linetype = 2, color = "grey50") +
    labs(x = indicator, y = "Odds ratio", title = paste("Restricted cubic spline:", indicator)) +
    theme_classic(base_size = 12)
  ggsave(file.path(figures_dir, "fig4_rcs.png"), p, width = 7, height = 5, dpi = 300)
  TRUE
}

subgroup_analysis <- function(design, data, indicator) {
  data$age_group <- ifelse(data$age >= 65, ">=65", "20-64")
  design$variables$age_group <- data$age_group
  subgroup_vars <- present(c("sex", "age_group", "diabetes", "hypertension"), data)
  covars_full <- present(c("age", "sex", "race_ethnicity", "education", "pir", "smoking_status", "alcohol_use", "physical_activity", "bmi", "hypertension", "diabetes", "total_cholesterol", "hdl_cholesterol", "triglycerides"), data)
  rows <- list()
  for (sg in subgroup_vars) {
    levels_sg <- unique(na.omit(as.character(design$variables[[sg]])))
    for (lvl in levels_sg) {
      sg_values <- as.character(design$variables[[sg]])
      dsub <- design[!is.na(sg_values) & sg_values == lvl, ]
      if (nrow(dsub$variables) < 50 || length(unique(dsub$variables$cvd)) < 2) next
      covars <- setdiff(covars_full, c(indicator, sg))
      fit <- tryCatch(svyglm(safe_formula("cvd", c(indicator, covars)), design = dsub, family = quasibinomial()), error = function(e) NULL)
      if (!is.null(fit)) {
        if (is.numeric(data[[indicator]])) {
          item <- tidy_or(fit, indicator, indicator, "Subgroup fully adjusted", indicator, paste(sg, lvl, sep = ": "))
        } else {
          item <- tidy_factor_or(fit, indicator, "Subgroup fully adjusted", paste(sg, lvl, sep = ": "))
        }
        if (!is.null(item)) {
          item$subgroup <- sg
          item$level <- lvl
          rows[[length(rows) + 1]] <- item
        }
      }
    }
  }
  bind_rows(rows)
}

plot_subgroup <- function(subgroup_df, figures_dir) {
  if (nrow(subgroup_df) == 0) return(FALSE)
  plot_df <- subgroup_df %>% mutate(label = paste(subgroup, level, sep = " = ")) %>% arrange(OR)
  p <- ggplot(plot_df, aes(x = OR, y = reorder(label, OR))) +
    geom_vline(xintercept = 1, linetype = 2, color = "grey55") +
    geom_errorbar(aes(xmin = CI_low, xmax = CI_high), width = 0.2, orientation = "y", color = "#4E79A7") +
    geom_point(size = 2.4, color = "#1F4E79") +
    scale_x_log10() +
    labs(x = "Odds ratio per 1-unit increase", y = NULL, title = unique(plot_df$indicator)[1]) +
    theme_classic(base_size = 12)
  ggsave(file.path(figures_dir, "fig5_subgroup_forest.png"), p, width = 7, height = 5, dpi = 300)
  TRUE
}

run_sensitivity <- function(data, weight_col, indicators) {
  if (!("combined_fasting_weight" %in% names(data)) || all(is.na(data$combined_fasting_weight))) return(data.frame())
  fasting_indicators <- intersect(indicators, c("tyg_index", "tyg_bmi", "tyg_wc", "aip"))
  if (length(fasting_indicators) == 0) return(data.frame())
  dfast <- data %>% filter(!is.na(combined_fasting_weight), combined_fasting_weight > 0)
  if (nrow(dfast) < 100) return(data.frame())
  design_fast <- make_design(dfast, "combined_fasting_weight")
  fit_indicator_models(design_fast, dfast, fasting_indicators) %>% mutate(sensitivity = "fasting_subsample_weight")
}

log_msg("Reading", input_path)
dat <- suppressMessages(read_csv(input_path, show_col_types = FALSE))
names(dat) <- make.names(names(dat))

required <- c("cvd", "strata", "psu", opt$weight)
missing_required <- setdiff(required, names(dat))
if (length(missing_required) > 0) stop("Missing required survey columns: ", paste(missing_required, collapse = ", "))

factor_vars <- present(c("sex", "race_ethnicity", "education", "smoking_status", "alcohol_use", "physical_activity", "hypertension", "diabetes"), dat)
for (v in factor_vars) dat[[v]] <- as.factor(dat[[v]])
dat$cvd <- as.numeric(dat$cvd)
dat <- dat %>% filter(!is.na(cvd), !is.na(.data[[opt$weight]]), .data[[opt$weight]] > 0, !is.na(strata), !is.na(psu))

design <- make_design(dat, opt$weight)
log_msg("Survey analytic rows:", nrow(dat))

table1 <- make_table1(design, dat)
write_csv(table1, file.path(tables_dir, "table1_baseline.csv"))
log_msg("Saved Table 1")

top_indicators <- read_top_indicators(dat, ranking_path, opt$top_n)
if (length(top_indicators) == 0) stop("No usable top indicators found for survey regression.")
log_msg("Top indicators:", paste(top_indicators, collapse = ", "))

regression <- fit_indicator_models(design, dat, top_indicators)
write_csv(regression, file.path(tables_dir, "table4_weighted_regression.csv"))
log_msg("Saved weighted regression table")

rcs_indicator <- ifelse(is.na(opt$rcs_indicator), top_indicators[1], opt$rcs_indicator)
if (rcs_indicator %in% names(dat)) {
  ok <- plot_rcs(design, dat, rcs_indicator, figures_dir)
  log_msg("RCS plot status:", ok)
} else {
  log_msg("RCS plot status: FALSE")
}

subgroup <- subgroup_analysis(design, dat, top_indicators[1])
write_csv(subgroup, file.path(tables_dir, "subgroup_analysis.csv"))
invisible(plot_subgroup(subgroup, figures_dir))
log_msg("Saved subgroup analysis")

sensitivity_indicators <- unique(c(top_indicators, present(c("tyg_index", "tyg_bmi", "tyg_wc", "aip"), dat)))
sensitivity <- run_sensitivity(dat, opt$weight, sensitivity_indicators)
if (nrow(sensitivity) > 0) {
  write_csv(sensitivity, file.path(tables_dir, "sensitivity_analysis.csv"))
  log_msg("Saved sensitivity analysis")
} else {
  log_msg("No sensitivity analysis generated")
}
log_msg("Survey analysis complete")
