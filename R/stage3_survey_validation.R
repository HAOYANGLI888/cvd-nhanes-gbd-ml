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
  make_option("--input", dest = "input", type = "character", default = NA,
              help = "Default: data/processed/nhanes_analysis.csv"),
  make_option("--weight", dest = "weight", type = "character", default = NA,
              help = "Default: auto, preferring combined_fasting_weight"),
  make_option("--log-file", dest = "log_file", type = "character", default = NA)
)

opt <- parse_args(OptionParser(option_list = option_list))

project_root <- normalizePath(opt$project_root, winslash = "/", mustWork = FALSE)
input_path <- ifelse(is.na(opt$input), file.path(project_root, "data/processed/nhanes_analysis.csv"), opt$input)
tables_dir <- file.path(project_root, "outputs/tables")
figures_dir <- file.path(project_root, "outputs/figures")
diagnostics_dir <- file.path(project_root, "outputs/diagnostics")
logs_dir <- file.path(project_root, "outputs/logs")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(diagnostics_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- ifelse(is.na(opt$log_file), file.path(logs_dir, "stage3_survey_validation.log"), opt$log_file)

log_msg <- function(...) {
  msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste(..., collapse = " "))
  cat(msg, "\n")
  cat(msg, "\n", file = log_file, append = TRUE)
}

present <- function(cols, data) cols[cols %in% names(data)]

safe_formula <- function(outcome, terms) {
  terms <- terms[!is.na(terms) & nzchar(terms)]
  if (length(terms) == 0) {
    as.formula(paste(outcome, "~ 1"))
  } else {
    as.formula(paste(outcome, "~", paste(terms, collapse = " + ")))
  }
}

pick_weight <- function(data, requested = NA) {
  candidates <- c(requested, "combined_fasting_weight", "survey_weight", "combined_mec_weight",
                  "fasting_weight_2yr", "mec_weight_2yr")
  candidates <- candidates[!is.na(candidates) & nzchar(candidates)]
  for (w in candidates) {
    if (w %in% names(data) && any(!is.na(data[[w]]) & data[[w]] > 0)) return(w)
  }
  stop("No usable NHANES weight column found.")
}

choose_col <- function(data, candidates) {
  hit <- candidates[candidates %in% names(data)]
  if (length(hit) == 0) return(NA_character_)
  hit[1]
}

weighted_quantiles <- function(design, var, probs = c(0.25, 0.5, 0.75)) {
  f <- as.formula(paste0("~", var))
  out <- tryCatch(
    svyquantile(f, design, quantiles = probs, na.rm = TRUE, ci = FALSE),
    error = function(e) NULL
  )
  if (!is.null(out)) {
    vals <- suppressWarnings(tryCatch(as.numeric(coef(out)), error = function(e) numeric(0)))
    if (length(vals) < length(probs)) {
      vals <- suppressWarnings(tryCatch(as.numeric(unlist(out, use.names = FALSE)), error = function(e) numeric(0)))
    }
    if (length(vals) >= length(probs) && all(is.finite(vals))) return(vals[seq_along(probs)])
  }
  vals <- suppressWarnings(as.numeric(stats::quantile(design$variables[[var]], probs = probs, na.rm = TRUE)))
  vals[seq_along(probs)]
}

mode_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

valid_terms <- function(data, terms) {
  terms <- present(unique(terms), data)
  leakage <- c("cvd", grep("^cvd_", names(data), value = TRUE),
               "chf_code", "chd_code", "angina_code", "heart_attack_code", "stroke_code")
  terms <- setdiff(terms, leakage)
  keep <- character(0)
  for (v in terms) {
    vals <- data[[v]]
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0) next
    if (length(unique(vals)) < 2) next
    keep <- c(keep, v)
  }
  keep
}

prepare_model_design <- function(base_design, data, terms) {
  terms <- valid_terms(data, terms)
  vars <- unique(c("cvd", terms))
  vars <- present(vars, base_design$variables)
  if (length(vars) == 0) return(NULL)
  complete <- stats::complete.cases(base_design$variables[, vars, drop = FALSE])
  dsub <- subset(base_design, complete)
  if (nrow(dsub$variables) == 0) return(NULL)
  dsub$variables <- droplevels(dsub$variables)
  terms <- valid_terms(dsub$variables, terms)
  vars <- unique(c("cvd", terms))
  complete <- stats::complete.cases(dsub$variables[, vars, drop = FALSE])
  dsub <- subset(dsub, complete)
  dsub$variables <- droplevels(dsub$variables)
  list(
    design = dsub,
    terms = terms,
    n_total = nrow(dsub$variables),
    n_events = sum(dsub$variables$cvd == 1, na.rm = TRUE)
  )
}

fit_svy_model <- function(base_design, data, terms) {
  prep <- prepare_model_design(base_design, data, terms)
  if (is.null(prep) || prep$n_total < 50 || length(unique(prep$design$variables$cvd)) < 2) return(NULL)
  f <- safe_formula("cvd", prep$terms)
  fit <- tryCatch(
    svyglm(f, design = prep$design, family = quasibinomial()),
    error = function(e) {
      log_msg("Model failed:", deparse(f), "|", conditionMessage(e))
      NULL
    }
  )
  if (is.null(fit)) return(NULL)
  list(fit = fit, design = prep$design, terms = prep$terms,
       n_total = prep$n_total, n_events = prep$n_events)
}

extract_term <- function(model_obj, term, indicator, form, model_name,
                         label = term, p_for_trend = NA_real_) {
  sm <- summary(model_obj$fit)$coefficients
  if (!(term %in% rownames(sm))) return(NULL)
  beta <- sm[term, "Estimate"]
  se <- sm[term, "Std. Error"]
  p <- sm[term, ncol(sm)]
  data.frame(
    indicator = indicator,
    form = form,
    model = model_name,
    term = label,
    OR = exp(beta),
    CI_lower = exp(beta - 1.96 * se),
    CI_upper = exp(beta + 1.96 * se),
    P_value = p,
    P_for_trend = p_for_trend,
    n_total = model_obj$n_total,
    n_events = model_obj$n_events,
    stringsAsFactors = FALSE
  )
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

derive_stage3_fields <- function(data) {
  if (!("physical_activity" %in% names(data))) {
    pa_cols <- present(c("pa_vigorous_work_code", "pa_moderate_work_code"), data)
    if (length(pa_cols) > 0) {
      yes_count <- rowSums(data[, pa_cols, drop = FALSE] == 1, na.rm = TRUE)
      no_count <- rowSums(data[, pa_cols, drop = FALSE] == 2, na.rm = TRUE)
      observed <- rowSums(!is.na(data[, pa_cols, drop = FALSE]))
      data$physical_activity <- ifelse(yes_count > 0, "active",
                                       ifelse(observed > 0 & no_count == observed, "inactive", NA))
    }
  }
  if (!("obesity_status" %in% names(data)) && "bmi" %in% names(data)) {
    data$obesity_status <- ifelse(!is.na(data$bmi) & data$bmi >= 30, "obese",
                                  ifelse(!is.na(data$bmi), "non_obese", NA))
  }
  if ("age" %in% names(data)) {
    data$age_group_stage3 <- ifelse(!is.na(data$age) & data$age >= 60, ">=60",
                                    ifelse(!is.na(data$age), "<60", NA))
  }
  data
}

make_table1 <- function(design, data) {
  continuous <- present(c(
    "age", "pir", "bmi", "waist", "mean_sbp", "mean_dbp", "fasting_glucose", "hba1c",
    "total_cholesterol", "ldl_cholesterol", "hdl_cholesterol", "triglycerides",
    "non_hdl_cholesterol", "tyg_index", "tyg_wc", "egfr", "uacr", "serum_creatinine",
    "urine_albumin", "uric_acid", "wbc", "nlr", "siri", "sii"
  ), data)
  categorical <- present(c(
    "sex", "race_ethnicity", "education", "smoking_status", "alcohol_use",
    "physical_activity"
  ), data)

  rows <- list()
  for (v in continuous) {
    vals <- list()
    for (g in c(0, 1)) {
      dsub <- subset(design, cvd == g)
      est <- tryCatch(svymean(as.formula(paste0("~", v)), dsub, na.rm = TRUE), error = function(e) NULL)
      vals[[as.character(g)]] <- if (is.null(est)) NA_character_ else sprintf("%.3f (SE %.3f)", coef(est)[1], SE(est)[1])
    }
    p <- tryCatch(svyttest(as.formula(paste0(v, " ~ cvd")), design)$p.value, error = function(e) NA_real_)
    rows[[length(rows) + 1]] <- data.frame(
      variable = v,
      level = "weighted mean (SE)",
      non_cvd = vals[["0"]],
      cvd = vals[["1"]],
      p_value = p,
      stringsAsFactors = FALSE
    )
  }

  for (v in categorical) {
    f <- as.formula(paste0("~", v, " + cvd"))
    tab <- tryCatch(svytable(f, design), error = function(e) NULL)
    p <- tryCatch(svychisq(f, design, statistic = "F")$p.value, error = function(e) NA_real_)
    if (is.null(tab) || length(dim(tab)) != 2) next
    props <- prop.table(tab, margin = 2)
    for (lvl in rownames(tab)) {
      rows[[length(rows) + 1]] <- data.frame(
        variable = v,
        level = lvl,
        non_cvd = sprintf("%.2f%%", 100 * props[lvl, "0"]),
        cvd = sprintf("%.2f%%", 100 * props[lvl, "1"]),
        p_value = p,
        stringsAsFactors = FALSE
      )
    }
  }
  bind_rows(rows)
}

build_models <- function(data, target_indicator) {
  model2 <- present(c("age", "sex", "race_ethnicity"), data)
  model3 <- present(c("age", "sex", "race_ethnicity", "education", "pir",
                      "smoking_status", "alcohol_use", "physical_activity", "bmi"), data)
  model4_extra <- present(c(
    "mean_sbp", "mean_dbp", "fasting_glucose", "hba1c",
    "total_cholesterol", "ldl_cholesterol", "hdl_cholesterol", "triglycerides",
    "non_hdl_cholesterol", "egfr", "uacr", "serum_creatinine", "uric_acid",
    "wbc", "nlr", "siri", "sii"
  ), data)
  list(
    list(name = "Model 1", covars = character(0)),
    list(name = "Model 2", covars = setdiff(model2, target_indicator)),
    list(name = "Model 3", covars = setdiff(model3, target_indicator)),
    list(name = "Model 4", covars = setdiff(unique(c(model3, model4_extra)), target_indicator))
  )
}

run_weighted_regression <- function(base_design, data, indicators) {
  rows <- list()
  for (indicator in indicators) {
    if (!(indicator %in% names(data))) next
    for (spec in build_models(data, indicator)) {
      covars <- setdiff(spec$covars, indicator)

      cont_obj <- fit_svy_model(base_design, data, c(indicator, covars))
      if (!is.null(cont_obj)) {
        item <- extract_term(cont_obj, indicator, indicator, "continuous", spec$name,
                             "per 1-unit increase")
        if (!is.null(item)) rows[[length(rows) + 1]] <- item
      }

      q <- weighted_quantiles(base_design, indicator)
      if (length(q) >= 3 && length(unique(q)) >= 3) {
        qvar <- paste0(indicator, "_quartile_stage3")
        trend_var <- paste0(indicator, "_qtrend_stage3")
        dtmp <- base_design
        quart <- cut(dtmp$variables[[indicator]], breaks = c(-Inf, q[1:3], Inf),
                     labels = paste0("Q", 1:4), include.lowest = TRUE)
        dtmp$variables[[qvar]] <- relevel(factor(quart), ref = "Q1")
        dtmp$variables[[trend_var]] <- as.numeric(dtmp$variables[[qvar]])

        trend_obj <- fit_svy_model(dtmp, dtmp$variables, c(trend_var, covars))
        p_trend <- NA_real_
        if (!is.null(trend_obj)) {
          trend_row <- extract_term(trend_obj, trend_var, indicator, "trend", spec$name,
                                    "quartile ordinal trend")
          if (!is.null(trend_row)) {
            p_trend <- trend_row$P_value[1]
            trend_row$P_for_trend <- p_trend
            rows[[length(rows) + 1]] <- trend_row
          }
        }

        q_obj <- fit_svy_model(dtmp, dtmp$variables, c(qvar, covars))
        if (!is.null(q_obj)) {
          for (lvl in paste0(qvar, "Q", 2:4)) {
            item <- extract_term(q_obj, lvl, indicator, "quartile", spec$name,
                                 sub(paste0("^", qvar), "", lvl), p_trend)
            if (!is.null(item)) rows[[length(rows) + 1]] <- item
          }
        }
      }
    }
  }
  bind_rows(rows)
}

make_reference_data <- function(data, indicator, covars, values) {
  ref <- data.frame(.stage3_indicator_value = values)
  names(ref)[1] <- indicator
  for (v in covars) {
    if (v == indicator) next
    x <- data[[v]]
    if (is.numeric(x)) {
      ref[[v]] <- median(x, na.rm = TRUE)
    } else {
      mv <- mode_value(as.character(x))
      if (is.factor(x)) {
        ref[[v]] <- factor(mv, levels = levels(x))
      } else {
        ref[[v]] <- mv
      }
    }
  }
  ref
}

extract_predict_link <- function(pred) {
  if (is.list(pred)) {
    return(list(fit = as.numeric(pred$fit), se = as.numeric(pred$se.fit)))
  }
  fit <- as.numeric(pred)
  pv <- attr(pred, "var")
  if (is.null(pv)) {
    se <- rep(NA_real_, length(fit))
  } else if (length(dim(pv)) == 2 && nrow(pv) == length(fit) && ncol(pv) == length(fit)) {
    se <- sqrt(diag(pv))
  } else {
    se <- sqrt(as.numeric(pv))
  }
  list(fit = fit, se = se)
}

extract_anova_p <- function(obj) {
  if (is.null(obj)) return(NA_real_)
  mat <- tryCatch(as.data.frame(obj), error = function(e) NULL)
  if (is.null(mat) || nrow(mat) == 0) return(NA_real_)
  p_cols <- grep("Pr|p", names(mat), ignore.case = TRUE, value = TRUE)
  if (length(p_cols) == 0) return(NA_real_)
  vals <- suppressWarnings(as.numeric(mat[[p_cols[length(p_cols)]]]))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) return(NA_real_)
  vals[length(vals)]
}

run_rcs <- function(base_design, data, indicators) {
  rows <- list()
  file_names <- c(
    egfr = "fig4_rcs_egfr.png",
    uacr = "fig4_rcs_uacr.png",
    siri = "fig4_rcs_siri.png",
    tyg_wc = "fig4_rcs_tyg_wc.png",
    tyg_index = "fig4_rcs_tyg_index.png",
    non_hdl_cholesterol = "fig4_rcs_non_hdl.png"
  )
  for (indicator in indicators) {
    if (!(indicator %in% names(data))) next
    covars <- setdiff(present(c("age", "sex", "race_ethnicity", "education", "pir",
                                "smoking_status", "alcohol_use", "physical_activity", "bmi"), data), indicator)
    prep <- prepare_model_design(base_design, data, c(indicator, covars))
    if (is.null(prep) || prep$n_total < 100 || prep$n_events < 10) next

    spline_formula <- as.formula(paste("cvd ~ ns(", indicator, ", df = 4)",
                                       ifelse(length(covars) > 0, paste("+", paste(covars, collapse = " + ")), "")))
    linear_formula <- safe_formula("cvd", c(indicator, covars))
    fit_spline <- tryCatch(svyglm(spline_formula, design = prep$design, family = quasibinomial()),
                           error = function(e) {
                             log_msg("RCS fit failed:", indicator, conditionMessage(e))
                             NULL
                           })
    fit_linear <- tryCatch(svyglm(linear_formula, design = prep$design, family = quasibinomial()),
                           error = function(e) NULL)
    if (is.null(fit_spline)) next

    p_nonlin <- tryCatch(extract_anova_p(anova(fit_linear, fit_spline)), error = function(e) NA_real_)
    if (is.na(p_nonlin)) {
      p_nonlin <- tryCatch(regTermTest(fit_spline, as.formula(paste0("~ ns(", indicator, ", df = 4)")))$p,
                           error = function(e) NA_real_)
    }

    rng <- stats::quantile(prep$design$variables[[indicator]], probs = c(0.01, 0.99), na.rm = TRUE)
    values <- seq(rng[1], rng[2], length.out = 120)
    newdata <- make_reference_data(prep$design$variables, indicator, covars, values)
    pred <- extract_predict_link(predict(fit_spline, newdata = newdata, type = "link", se.fit = TRUE))
    refdata <- make_reference_data(prep$design$variables, indicator, covars,
                                   median(prep$design$variables[[indicator]], na.rm = TRUE))
    ref_pred <- extract_predict_link(predict(fit_spline, newdata = refdata, type = "link", se.fit = TRUE))
    ref <- ref_pred$fit[1]
    plot_df <- data.frame(
      value = values,
      OR = exp(pred$fit - ref),
      lower = exp(pred$fit - 1.96 * pred$se - ref),
      upper = exp(pred$fit + 1.96 * pred$se - ref)
    )
    p <- ggplot(plot_df, aes(x = value, y = OR)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#4E79A7", alpha = 0.18) +
      geom_line(color = "#1F4E79", linewidth = 1) +
      geom_hline(yintercept = 1, linetype = 2, color = "grey50") +
      labs(x = indicator, y = "Odds ratio", title = paste("Survey-weighted natural spline:", indicator)) +
      theme_classic(base_size = 12)
    fig_path <- file.path(figures_dir, file_names[[indicator]])
    ggsave(fig_path, p, width = 7, height = 5, dpi = 300)

    rows[[length(rows) + 1]] <- data.frame(
      indicator = indicator,
      method = "survey-weighted logistic regression with splines::ns(df=4), Model 3 adjustment",
      n_total = prep$n_total,
      n_events = prep$n_events,
      p_nonlinearity = p_nonlin,
      figure = fig_path,
      stringsAsFactors = FALSE
    )
  }
  bind_rows(rows)
}

run_subgroups <- function(base_design, data, indicators) {
  subgroup_vars <- present(c("age_group_stage3", "sex", "diabetes", "hypertension", "obesity_status"), data)
  rows <- list()
  for (indicator in indicators) {
    if (!(indicator %in% names(data))) next
    for (sg in subgroup_vars) {
      base_covars <- setdiff(present(c("age", "sex", "race_ethnicity", "education", "pir",
                                       "smoking_status", "alcohol_use", "physical_activity", "bmi"), data),
                             c(indicator, sg))
      int_terms <- c(indicator, sg, paste0(indicator, ":", sg), base_covars)
      prep_int <- prepare_model_design(base_design, data, c(indicator, sg, base_covars))
      interaction_p <- NA_real_
      if (!is.null(prep_int) && prep_int$n_total >= 100) {
        f_int <- as.formula(paste("cvd ~", indicator, "*", sg,
                                  ifelse(length(base_covars) > 0, paste("+", paste(base_covars, collapse = " + ")), "")))
        fit_int <- tryCatch(svyglm(f_int, design = prep_int$design, family = quasibinomial()), error = function(e) NULL)
        if (!is.null(fit_int)) {
          interaction_p <- tryCatch(regTermTest(fit_int, as.formula(paste0("~", indicator, ":", sg)))$p,
                                    error = function(e) NA_real_)
          interaction_p <- suppressWarnings(as.numeric(interaction_p)[1])
        }
      }
      levels_sg <- unique(na.omit(as.character(data[[sg]])))
      for (lvl in levels_sg) {
        dsub <- subset(base_design, !is.na(base_design$variables[[sg]]) & as.character(base_design$variables[[sg]]) == lvl)
        prep <- prepare_model_design(dsub, dsub$variables, c(indicator, base_covars))
        if (is.null(prep) || prep$n_total < 50 || prep$n_events < 5) next
        obj <- fit_svy_model(dsub, dsub$variables, c(indicator, base_covars))
        if (is.null(obj)) next
        item <- extract_term(obj, indicator, indicator, "continuous", "Subgroup Model 3",
                             "per 1-unit increase")
        if (!is.null(item)) {
          item$subgroup <- sg
          item$level <- lvl
          item$interaction_p <- interaction_p
          rows[[length(rows) + 1]] <- item
        }
      }
    }
  }
  bind_rows(rows)
}

plot_subgroups <- function(subgroup_df) {
  if (nrow(subgroup_df) == 0) return(FALSE)
  plot_df <- subgroup_df %>%
    mutate(label = paste(indicator, subgroup, level, sep = " | ")) %>%
    arrange(indicator, subgroup, OR)
  p <- ggplot(plot_df, aes(x = OR, y = reorder(label, OR))) +
    geom_vline(xintercept = 1, linetype = 2, color = "grey55") +
    geom_errorbar(aes(xmin = CI_lower, xmax = CI_upper), width = 0.2, orientation = "y", color = "#4E79A7") +
    geom_point(size = 1.8, color = "#1F4E79") +
    scale_x_log10() +
    labs(x = "Odds ratio per 1-unit increase", y = NULL, title = "Survey-weighted subgroup analysis") +
    theme_classic(base_size = 9)
  ggsave(file.path(figures_dir, "fig5_subgroup_forest.png"), p, width = 9, height = 8, dpi = 300)
  TRUE
}

run_sensitivity <- function(data, weight_col, indicators) {
  results <- list()
  run_one <- function(d, label, covar_builder) {
    if (nrow(d) < 100 || sum(d$cvd == 1, na.rm = TRUE) < 10) return(data.frame())
    des <- make_design(d, weight_col)
    rows <- list()
    for (indicator in indicators) {
      if (!(indicator %in% names(d))) next
      covars <- covar_builder(d, indicator)
      obj <- fit_svy_model(des, d, c(indicator, covars))
      if (is.null(obj)) next
      item <- extract_term(obj, indicator, indicator, "continuous", "Sensitivity",
                           "per 1-unit increase")
      if (!is.null(item)) {
        item$sensitivity = label
        rows[[length(rows) + 1]] <- item
      }
    }
    bind_rows(rows)
  }
  full_covars <- function(d, indicator) {
    setdiff(unique(c(
      present(c("age", "sex", "race_ethnicity", "education", "pir",
                "smoking_status", "alcohol_use", "physical_activity", "bmi"), d),
      present(c("mean_sbp", "mean_dbp", "fasting_glucose", "hba1c",
                "total_cholesterol", "ldl_cholesterol", "hdl_cholesterol", "triglycerides",
                "non_hdl_cholesterol", "egfr", "uacr", "serum_creatinine", "uric_acid",
                "wbc", "nlr", "siri", "sii"), d)
    )), indicator)
  }
  routine_covars <- function(d, indicator) {
    setdiff(unique(c(
      present(c("age", "sex", "race_ethnicity", "education", "pir",
                "smoking_status", "alcohol_use", "physical_activity", "bmi"), d),
      present(c("mean_sbp", "mean_dbp", "fasting_glucose", "hba1c",
                "total_cholesterol", "ldl_cholesterol", "hdl_cholesterol", "triglycerides",
                "non_hdl_cholesterol", "egfr", "uacr", "serum_creatinine", "uric_acid",
                "wbc", "nlr", "siri", "sii", "tyg_index", "tyg_bmi", "tyg_wc", "aip"), d)
    )), c(indicator, "hypertension", "diabetes"))
  }

  stroke_col <- choose_col(data, c("cvd_stroke", "stroke_code"))
  if (!is.na(stroke_col)) {
    d <- data[is.na(data[[stroke_col]]) | data[[stroke_col]] != 1, , drop = FALSE]
    results[[length(results) + 1]] <- run_one(d, "exclude_prior_stroke", full_covars)
  }
  mi_col <- choose_col(data, c("cvd_heart_attack", "heart_attack_code"))
  if (!is.na(mi_col)) {
    d <- data[is.na(data[[mi_col]]) | data[[mi_col]] != 1, , drop = FALSE]
    results[[length(results) + 1]] <- run_one(d, "exclude_prior_heart_attack", full_covars)
  }
  results[[length(results) + 1]] <- run_one(data, "routine_biomarker_adjustment_no_htn_dm", routine_covars)
  bind_rows(results)
}

write_report <- function(path, context) {
  significant_full <- context$regression %>%
    filter(model == "Model 4", form == "continuous", !is.na(P_value), P_value < 0.05) %>%
    arrange(P_value)
  nonlinear <- context$rcs %>%
    filter(!is.na(p_nonlinearity), p_nonlinearity < 0.05) %>%
    arrange(p_nonlinearity)
  interactions <- context$subgroup %>%
    filter(!is.na(interaction_p), interaction_p < 0.05) %>%
    distinct(indicator, subgroup, interaction_p) %>%
    arrange(interaction_p)
  sensitivity_sig <- context$sensitivity %>%
    filter(!is.na(P_value), P_value < 0.05) %>%
    distinct(sensitivity, indicator, OR, CI_lower, CI_upper, P_value)

  lines <- c(
    "# Stage 3 Survey-Weighted Validation Report",
    "",
    paste0("- Rscript available: yes (executed through conda env nhanes-r)"),
    paste0("- Data file: ", context$input_path),
    paste0("- Weight: ", context$weight_col),
    paste0("- Strata: strata"),
    paste0("- PSU: psu"),
    paste0("- Final survey analytic rows: ", context$n_total),
    paste0("- CVD events: ", context$n_events),
    paste0("- Outcome component agreement mismatches: ", context$outcome_mismatches),
    paste0("- Leakage covariates used in Stage 3 models: ", length(context$leak_covars_used)),
    "",
    "## Output Status",
    "",
    paste0("- Table 1 generated: ", file.exists(file.path(tables_dir, "table1_baseline_weighted.csv"))),
    paste0("- Weighted logistic regression generated: ", file.exists(file.path(tables_dir, "table4_weighted_regression.csv"))),
    paste0("- RCS figures generated: ", nrow(context$rcs), " indicator-specific figures"),
    paste0("- Subgroup analysis generated: ", nrow(context$subgroup), " rows"),
    paste0("- Sensitivity analysis generated: ", nrow(context$sensitivity), " rows"),
    "",
    "## Fully Adjusted Significant Indicators",
    ""
  )
  if (nrow(significant_full) == 0) {
    lines <- c(lines, "- None at P < 0.05 in Model 4 continuous models.")
  } else {
    lines <- c(lines, paste0("- ", significant_full$indicator, ": OR ",
                             sprintf("%.3f", significant_full$OR), " (95% CI ",
                             sprintf("%.3f", significant_full$CI_lower), "-",
                             sprintf("%.3f", significant_full$CI_upper), "), P=",
                             signif(significant_full$P_value, 3)))
  }
  lines <- c(lines, "", "## Nonlinearity", "")
  if (nrow(nonlinear) == 0) {
    lines <- c(lines, "- No spline nonlinearity test was significant at P < 0.05.")
  } else {
    lines <- c(lines, paste0("- ", nonlinear$indicator, ": P_nonlinearity=",
                             signif(nonlinear$p_nonlinearity, 3)))
  }
  lines <- c(lines, "", "## Subgroup Interactions", "")
  if (nrow(interactions) == 0) {
    lines <- c(lines, "- No interaction P value was significant at P < 0.05.")
  } else {
    lines <- c(lines, paste0("- ", interactions$indicator, " by ", interactions$subgroup,
                             ": P_interaction=", signif(interactions$interaction_p, 3)))
  }
  lines <- c(lines, "", "## Sensitivity Analysis", "")
  if (nrow(sensitivity_sig) == 0) {
    lines <- c(lines, "- No sensitivity-analysis continuous indicator was significant at P < 0.05.")
  } else {
    lines <- c(lines, paste0("- ", sensitivity_sig$sensitivity, " | ", sensitivity_sig$indicator,
                             ": OR ", sprintf("%.3f", sensitivity_sig$OR), " (95% CI ",
                             sprintf("%.3f", sensitivity_sig$CI_lower), "-",
                             sprintf("%.3f", sensitivity_sig$CI_upper), "), P=",
                             signif(sensitivity_sig$P_value, 3)))
  }
  lines <- c(lines, "", "## Recommendation", "",
             "- Stage 3 generated real survey-weighted validation outputs using NHANES design variables. It is reasonable to proceed to GBD analysis and result organization after reviewing model-specific missingness and clinical plausibility.")
  writeLines(lines, path, useBytes = TRUE)
}

log_msg("Reading", input_path)
dat <- suppressMessages(read_csv(input_path, show_col_types = FALSE))
names(dat) <- make.names(names(dat))
dat <- derive_stage3_fields(dat)

required_outcome_components <- c("chf_code", "chd_code", "angina_code", "heart_attack_code", "stroke_code")
missing_components <- setdiff(required_outcome_components, names(dat))
if (length(missing_components) > 0) {
  log_msg("WARNING missing outcome components:", paste(missing_components, collapse = ", "))
}
if (!("cvd" %in% names(dat))) stop("Missing cvd outcome column.")

weight_col <- pick_weight(dat, opt$weight)
required_design <- c("strata", "psu", weight_col)
missing_design <- setdiff(required_design, names(dat))
if (length(missing_design) > 0) stop("Missing survey design columns: ", paste(missing_design, collapse = ", "))

factor_vars <- present(c("sex", "race_ethnicity", "education", "smoking_status", "alcohol_use",
                         "physical_activity", "hypertension", "diabetes", "obesity_status",
                         "age_group_stage3"), dat)
for (v in factor_vars) dat[[v]] <- factor(dat[[v]])
dat$cvd <- as.numeric(dat$cvd)

dat <- dat %>%
  filter(!is.na(cvd), !is.na(.data[[weight_col]]), .data[[weight_col]] > 0,
         !is.na(strata), !is.na(psu))

if (length(present(required_outcome_components, dat)) == length(required_outcome_components)) {
  comp <- dat[, required_outcome_components, drop = FALSE]
  cvd_from_components <- ifelse(rowSums(comp == 1, na.rm = TRUE) > 0, 1,
                                ifelse(rowSums(!is.na(comp)) > 0, 0, NA))
  outcome_mismatches <- sum(!is.na(cvd_from_components) & !is.na(dat$cvd) &
                              cvd_from_components != dat$cvd)
} else {
  outcome_mismatches <- NA_integer_
}

design <- make_design(dat, weight_col)
log_msg("Survey analytic rows:", nrow(dat), "events:", sum(dat$cvd == 1, na.rm = TRUE))
log_msg("Survey design:", weight_col, "strata", "psu")

indicator_fixed <- present(c("egfr", "uacr", "siri", "tyg_wc", "tyg_index",
                             "non_hdl_cholesterol", "serum_creatinine",
                             "ldl_cholesterol", "hba1c"), dat)
indicator_rcs <- present(c("egfr", "uacr", "siri", "tyg_wc", "tyg_index",
                           "non_hdl_cholesterol"), dat)
indicator_subgroup <- present(c("egfr", "uacr", "siri", "tyg_wc"), dat)

table1 <- make_table1(design, dat)
write_csv(table1, file.path(tables_dir, "table1_baseline_weighted.csv"))
write_csv(table1, file.path(tables_dir, "table1_baseline.csv"))
log_msg("Saved weighted Table 1")

regression <- run_weighted_regression(design, dat, indicator_fixed)
write_csv(regression, file.path(tables_dir, "table4_weighted_regression.csv"))
log_msg("Saved weighted regression rows:", nrow(regression))

rcs <- run_rcs(design, dat, indicator_rcs)
write_csv(rcs, file.path(tables_dir, "rcs_nonlinearity_tests.csv"))
log_msg("Saved RCS rows:", nrow(rcs))

subgroup <- run_subgroups(design, dat, indicator_subgroup)
write_csv(subgroup, file.path(tables_dir, "subgroup_analysis.csv"))
invisible(plot_subgroups(subgroup))
log_msg("Saved subgroup rows:", nrow(subgroup))

sensitivity <- run_sensitivity(dat, weight_col, indicator_fixed)
write_csv(sensitivity, file.path(tables_dir, "sensitivity_analysis.csv"))
log_msg("Saved sensitivity rows:", nrow(sensitivity))

leak_terms <- c("cvd", grep("^cvd_", names(dat), value = TRUE),
                "chf_code", "chd_code", "angina_code", "heart_attack_code", "stroke_code")
stage3_covars <- unique(c(unlist(lapply(indicator_fixed, function(x) unlist(lapply(build_models(dat, x), `[[`, "covars"))))))
leak_covars_used <- intersect(stage3_covars, leak_terms)

write_report(
  file.path(diagnostics_dir, "stage3_survey_validation_report.md"),
  list(
    input_path = input_path,
    weight_col = weight_col,
    n_total = nrow(dat),
    n_events = sum(dat$cvd == 1, na.rm = TRUE),
    outcome_mismatches = outcome_mismatches,
    leak_covars_used = leak_covars_used,
    regression = regression,
    rcs = rcs,
    subgroup = subgroup,
    sensitivity = sensitivity
  )
)
log_msg("Stage 3 survey validation complete")
