suppressPackageStartupMessages({
  library(optparse)
  library(survey)
})

option_list <- list(
  make_option(c("--project-root"), type = "character", default = "."),
  make_option(c("--input"), type = "character", default = NA),
  make_option(c("--output"), type = "character", default = NA)
)
opt <- parse_args(OptionParser(option_list = option_list))

root <- normalizePath(opt$`project-root`, winslash = "/", mustWork = TRUE)
input_path <- ifelse(is.na(opt$input), file.path(root, "data/processed/nhanes_2021_2023_external.csv"), opt$input)
output_path <- ifelse(is.na(opt$output), file.path(root, "outputs/external_validation/external_weighted_regression_2021_2023.csv"), opt$output)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

options(survey.lonely.psu = "adjust")

if (!file.exists(input_path)) {
  stop(paste("External NHANES file not found:", input_path))
}

data <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)

required <- c("cvd", "strata", "psu")
missing_required <- setdiff(required, names(data))
if (length(missing_required) > 0) {
  stop(paste("Missing required design/outcome variables:", paste(missing_required, collapse = ", ")))
}

weight_candidates <- c("combined_fasting_weight", "survey_weight", "fasting_weight_2yr", "mec_weight_2yr")
weight_var <- weight_candidates[weight_candidates %in% names(data)][1]
if (is.na(weight_var)) {
  stop("No usable survey weight variable found.")
}

data[[weight_var]] <- suppressWarnings(as.numeric(data[[weight_var]]))
data <- data[!is.na(data[[weight_var]]) & data[[weight_var]] > 0 & !is.na(data$cvd), , drop = FALSE]
data$cvd <- as.numeric(data$cvd)
data$strata <- as.numeric(data$strata)
data$psu <- as.numeric(data$psu)

for (nm in intersect(c("sex", "race_ethnicity", "education", "smoking_status"), names(data))) {
  data[[nm]] <- as.factor(data[[nm]])
}

design <- svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = as.formula(paste0("~", weight_var)),
  data = data,
  nest = TRUE
)

indicators <- c("uacr", "egfr", "tyg_wc", "siri")
models <- list(
  "External Model A" = character(0),
  "External Model B" = c("age", "sex", "race_ethnicity"),
  "External Model C" = c("age", "sex", "race_ethnicity", "education", "pir", "smoking_status", "bmi")
)

fit_one <- function(indicator, model_name, covars) {
  if (!(indicator %in% names(data))) {
    return(data.frame(
      indicator = indicator, model = model_name, OR = NA_real_, CI_lower = NA_real_, CI_upper = NA_real_,
      P_value = NA_real_, n_total = 0, n_events = 0, direction = "indicator_missing",
      weight = weight_var, p_method = "not_available", stringsAsFactors = FALSE
    ))
  }
  available_covars <- covars[covars %in% names(data)]
  vars <- unique(c("cvd", indicator, available_covars, "strata", "psu", weight_var))
  complete <- complete.cases(data[, vars, drop = FALSE])
  dsub <- data[complete, , drop = FALSE]
  if (nrow(dsub) < 50 || length(unique(dsub$cvd)) < 2 || sum(dsub$cvd == 1, na.rm = TRUE) < 5) {
    return(data.frame(
      indicator = indicator, model = model_name, OR = NA_real_, CI_lower = NA_real_, CI_upper = NA_real_,
      P_value = NA_real_, n_total = nrow(dsub), n_events = sum(dsub$cvd == 1, na.rm = TRUE),
      direction = "insufficient_data", weight = weight_var, p_method = "not_available", stringsAsFactors = FALSE
    ))
  }
  fml <- paste("cvd ~", indicator)
  if (length(available_covars) > 0) {
    fml <- paste(fml, "+", paste(available_covars, collapse = " + "))
  }
  dsgn <- svydesign(
    ids = ~psu,
    strata = ~strata,
    weights = as.formula(paste0("~", weight_var)),
    data = dsub,
    nest = TRUE
  )
  fit <- tryCatch(svyglm(as.formula(fml), design = dsgn, family = quasibinomial()), error = function(e) NULL)
  if (is.null(fit) || !(indicator %in% rownames(coef(summary(fit))))) {
    return(data.frame(
      indicator = indicator, model = model_name, OR = NA_real_, CI_lower = NA_real_, CI_upper = NA_real_,
      P_value = NA_real_, n_total = nrow(dsub), n_events = sum(dsub$cvd == 1, na.rm = TRUE),
      direction = "fit_failed", weight = weight_var, p_method = "not_available", stringsAsFactors = FALSE
    ))
  }
  beta <- coef(fit)[indicator]
  se <- sqrt(vcov(fit)[indicator, indicator])
  p <- coef(summary(fit))[indicator, "Pr(>|t|)"]
  p_method <- "survey_t"
  if (is.na(p) && !is.na(beta) && !is.na(se) && se > 0) {
    p <- 2 * pnorm(abs(beta / se), lower.tail = FALSE)
    p_method <- "normal_approximation_when_survey_p_missing"
  }
  or <- exp(beta)
  lo <- exp(beta - 1.96 * se)
  hi <- exp(beta + 1.96 * se)
  direction <- ifelse(is.na(or), "not_available", ifelse(or > 1, "positive", ifelse(or < 1, "negative", "null")))
  data.frame(
    indicator = indicator,
    model = model_name,
    OR = as.numeric(or),
    CI_lower = as.numeric(lo),
    CI_upper = as.numeric(hi),
    P_value = as.numeric(p),
    n_total = nrow(dsub),
    n_events = sum(dsub$cvd == 1, na.rm = TRUE),
    direction = direction,
    weight = weight_var,
    p_method = p_method,
    stringsAsFactors = FALSE
  )
}

rows <- list()
idx <- 1
for (indicator in indicators) {
  for (model_name in names(models)) {
    rows[[idx]] <- fit_one(indicator, model_name, models[[model_name]])
    idx <- idx + 1
  }
}

out <- do.call(rbind, rows)
write.csv(out, output_path, row.names = FALSE)
cat("Wrote", output_path, "with", nrow(out), "rows\n")
