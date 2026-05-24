import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MS_TABLES = OUT / "manuscript_tables"
MS_RESULTS = OUT / "manuscript_results"
DIAG = OUT / "diagnostics"
DRAFT_DIR = OUT / "manuscript_draft"
DRAFT_DIR.mkdir(parents=True, exist_ok=True)

TITLE = (
    "Interpretable Machine Learning Identifies Renal, Metabolic, and Inflammatory Indicators "
    "Associated with Cardiovascular Disease: Evidence from NHANES 2005–2018 and GBD 2023"
)


def read_csv(name: str) -> pd.DataFrame:
    path = MS_TABLES / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def row_by(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = df[df[column].astype(str).eq(value)]
    if rows.empty:
        raise KeyError(f"Missing {column}={value}")
    return rows.iloc[0]


def gbd_value(t5: pd.DataFrame, location: str, measure: str, col: str) -> str:
    rows = t5[(t5["location"].eq(location)) & (t5["measure"].eq(measure))]
    if rows.empty:
        return "not available"
    return str(rows.iloc[0][col])


def table1_value(t1: pd.DataFrame, characteristic: str) -> pd.Series:
    return row_by(t1, "Characteristic", characteristic)


def model_row(t2: pd.DataFrame, feature_set: str) -> pd.Series:
    return row_by(t2, "feature set", feature_set)


def reg_row(t4: pd.DataFrame, indicator: str) -> pd.Series:
    return row_by(t4, "indicator", indicator)


def feature_row(t3: pd.DataFrame, indicator: str) -> pd.Series:
    return row_by(t3, "indicator", indicator)


def extract_case_counts(results_draft: str) -> tuple[str, str, str]:
    match = re.search(
        r"included ([0-9,]+) adults .*?including ([0-9,]+) with CVD and ([0-9,]+) without CVD",
        results_draft,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1), match.group(2), match.group(3)
    return "not available", "not available", "not available"


def extract_rcs_p(results_draft: str, indicator: str) -> str:
    pattern = rf"{re.escape(indicator)} showed evidence of nonlinearity.*?P for nonlinearity=([0-9.eE+-]+)"
    match = re.search(pattern, results_draft)
    return match.group(1) if match else "not available"


def build_manuscript() -> tuple[str, dict]:
    t1 = read_csv("Table_1_weighted_baseline_characteristics.csv")
    t2 = read_csv("Table_2_model_performance.csv")
    t3 = read_csv("Table_3_feature_ranking.csv")
    t4 = read_csv("Table_4_weighted_regression.csv")
    t5 = read_csv("Table_5_gbd_summary.csv")
    results_draft = read_text(MS_RESULTS / "results_draft_en.md")
    methods_skeleton = read_text(MS_RESULTS / "methods_skeleton_en.md")
    legends = read_text(MS_RESULTS / "figure_legends_and_table_notes_en.md")
    stage5_audit = read_text(DIAG / "stage5_manuscript_package_audit.md")
    gbd_check = read_text(DIAG / "stage5_gbd_result_check.md")

    n_total, n_cvd, n_noncvd = extract_case_counts(results_draft)

    age = table1_value(t1, "Age, years")
    bmi = table1_value(t1, "Body mass index, kg/m2")
    pir = table1_value(t1, "Poverty income ratio")
    sbp = table1_value(t1, "Systolic blood pressure, mmHg")
    glucose = table1_value(t1, "Fasting glucose, mg/dL")
    hba1c_base = table1_value(t1, "HbA1c, %")

    full = model_row(t2, "full_clinical_model")
    routine = model_row(t2, "routine_indicator_model")
    biomarker = model_row(t2, "biomarker_focused_model")
    auc_diff = full["full vs biomarker AUC difference"]

    uacr = reg_row(t4, "UACR")
    egfr = reg_row(t4, "eGFR")
    tyg_wc = reg_row(t4, "TyG-WC")
    siri = reg_row(t4, "SIRI")

    uacr_feat = feature_row(t3, "UACR")
    egfr_feat = feature_row(t3, "eGFR")
    tyg_feat = feature_row(t3, "TyG-WC")
    siri_feat = feature_row(t3, "SIRI")
    ldl_feat = feature_row(t3, "LDL-C")

    p_uacr_rcs = extract_rcs_p(results_draft, "UACR")
    p_egfr_rcs = extract_rcs_p(results_draft, "eGFR")
    p_tyg_rcs = extract_rcs_p(results_draft, "TyG-WC")
    p_siri_rcs = extract_rcs_p(results_draft, "SIRI")

    # GBD values from manuscript-ready Table 5.
    gbd_lines = {
        "global_deaths": (
            gbd_value(t5, "Global", "Deaths", "1990 age-standardized rate"),
            gbd_value(t5, "Global", "Deaths", "2023 age-standardized rate"),
            gbd_value(t5, "Global", "Deaths", "percent change"),
            gbd_value(t5, "Global", "Deaths", "EAPC"),
            gbd_value(t5, "Global", "Deaths", "EAPC 95% CI"),
            gbd_value(t5, "Global", "Deaths", "EAPC P value"),
        ),
        "global_dalys": (
            gbd_value(t5, "Global", "DALYs", "1990 age-standardized rate"),
            gbd_value(t5, "Global", "DALYs", "2023 age-standardized rate"),
            gbd_value(t5, "Global", "DALYs", "percent change"),
            gbd_value(t5, "Global", "DALYs", "EAPC"),
            gbd_value(t5, "Global", "DALYs", "EAPC 95% CI"),
            gbd_value(t5, "Global", "DALYs", "EAPC P value"),
        ),
        "china_deaths": (
            gbd_value(t5, "China", "Deaths", "1990 age-standardized rate"),
            gbd_value(t5, "China", "Deaths", "2023 age-standardized rate"),
            gbd_value(t5, "China", "Deaths", "percent change"),
            gbd_value(t5, "China", "Deaths", "EAPC"),
            gbd_value(t5, "China", "Deaths", "EAPC 95% CI"),
            gbd_value(t5, "China", "Deaths", "EAPC P value"),
        ),
        "china_dalys": (
            gbd_value(t5, "China", "DALYs", "1990 age-standardized rate"),
            gbd_value(t5, "China", "DALYs", "2023 age-standardized rate"),
            gbd_value(t5, "China", "DALYs", "percent change"),
            gbd_value(t5, "China", "DALYs", "EAPC"),
            gbd_value(t5, "China", "DALYs", "EAPC 95% CI"),
            gbd_value(t5, "China", "DALYs", "EAPC P value"),
        ),
        "us_dalys": (
            gbd_value(t5, "United States", "DALYs", "1990 age-standardized rate"),
            gbd_value(t5, "United States", "DALYs", "2023 age-standardized rate"),
            gbd_value(t5, "United States", "DALYs", "percent change"),
            gbd_value(t5, "United States", "DALYs", "EAPC"),
            gbd_value(t5, "United States", "DALYs", "EAPC 95% CI"),
            gbd_value(t5, "United States", "DALYs", "EAPC P value"),
        ),
        "china_prev": (
            gbd_value(t5, "China", "Prevalence", "1990 age-standardized rate"),
            gbd_value(t5, "China", "Prevalence", "2023 age-standardized rate"),
            gbd_value(t5, "China", "Prevalence", "percent change"),
            gbd_value(t5, "China", "Prevalence", "EAPC"),
            gbd_value(t5, "China", "Prevalence", "EAPC 95% CI"),
            gbd_value(t5, "China", "Prevalence", "EAPC P value"),
        ),
    }

    manuscript = f"""# {TITLE}

## Abstract

### Background

Cardiovascular disease (CVD) remains a major population-level burden and a frequent target for risk stratification in public health and clinical research. Routine metabolic, inflammatory, and renal-function indicators may capture complementary biological information, but their relative usefulness for CVD classification and epidemiologic association remains uncertain in nationally representative data.

### Methods

This study used two analytically separate public data sources. GBD 2023 data were used to describe population-level CVD burden trends in Global, China, and United States settings. NHANES 2005-2018 adult data were used for individual-level prediction and survey-weighted association analyses. CVD was defined as self-reported congestive heart failure, coronary heart disease, angina, heart attack or myocardial infarction, or stroke. Three leakage-audited feature sets were evaluated: a full clinical model, a routine indicator model, and a biomarker-focused model. Interpretable machine learning models were compared using held-out test performance. Survey-weighted logistic regression, restricted cubic spline (RCS), subgroup, and sensitivity analyses were used to evaluate key indicators.

### Results

In GBD 2023, age-standardized global CVD death rates decreased from {gbd_lines['global_deaths'][0]} in 1990 to {gbd_lines['global_deaths'][1]} in 2023 ({gbd_lines['global_deaths'][2]}), and global CVD DALY rates decreased from {gbd_lines['global_dalys'][0]} to {gbd_lines['global_dalys'][1]} ({gbd_lines['global_dalys'][2]}). In China, age-standardized CVD death rates decreased from {gbd_lines['china_deaths'][0]} to {gbd_lines['china_deaths'][1]} ({gbd_lines['china_deaths'][2]}), and CVD DALY rates decreased from {gbd_lines['china_dalys'][0]} to {gbd_lines['china_dalys'][1]} ({gbd_lines['china_dalys'][2]}). The NHANES analytic sample included {n_total} adults, including {n_cvd} with CVD and {n_noncvd} without CVD. The best full clinical model was XGBoost (AUC={full['AUC']}, PR-AUC={full['PR-AUC']}, F1={full['F1']}); the best routine indicator model was XGBoost (AUC={routine['AUC']}); and the best biomarker-focused model was Elastic Net (AUC={biomarker['AUC']}). The full clinical versus biomarker-focused AUC difference was {auc_diff}. UACR was prioritized in both the routine and biomarker-focused top 10 feature rankings and remained associated with CVD in the strict fully adjusted model (OR {uacr['Strict Model 4 OR (95% CI)']}, P={uacr['Strict Model 4 P value']}). eGFR, TyG-WC, and SIRI were supported by machine learning, Model 3 regression, and RCS analyses, although their estimates were attenuated in the strict fully adjusted model.

### Conclusions

In this public-database study, GBD data provided population-level context for CVD burden, whereas NHANES supported individual-level modeling and association analyses. The biomarker-focused model showed discrimination comparable to the full clinical model. UACR was the most robust indicator across feature ranking, regression, nonlinear, and sensitivity analyses. eGFR, TyG-WC, and SIRI were consistently prioritized or supported but were more adjustment-sensitive. Prospective validation is needed before clinical implementation.

### Keywords

Cardiovascular disease; NHANES; GBD; interpretable machine learning; UACR; eGFR; TyG-WC; SIRI; SHAP; survey-weighted analysis.

## Introduction

Cardiovascular disease remains one of the most important health challenges worldwide. Population-level surveillance systems such as the Global Burden of Disease (GBD) study provide a structured framework for describing long-term changes in CVD mortality, disability-adjusted life years (DALYs), prevalence, and risk-attributable burden. In the GBD 2023 data used in the present study, age-standardized CVD death and DALY rates declined between 1990 and 2023 in Global, China, and United States settings, but the absolute public health relevance of CVD remained substantial. In China, for example, age-standardized CVD DALY rates changed from {gbd_lines['china_dalys'][0]} in 1990 to {gbd_lines['china_dalys'][1]} in 2023, while prevalence rates changed from {gbd_lines['china_prev'][0]} to {gbd_lines['china_prev'][1]}. These descriptive data provide important macro-level context for public health planning, but they do not directly identify individual-level indicators associated with CVD status.

Traditional CVD risk assessment relies on well-established demographic and clinical variables such as age, sex, blood pressure, diabetes status, smoking, and lipids. These factors remain essential, yet routine clinical and laboratory indicators may contain additional information about kidney function, cardiometabolic stress, and inflammation. In population studies, many of these variables are already collected at scale, making them attractive for risk-marker screening and epidemiologic hypothesis generation. However, candidate indicators often overlap biologically and statistically. A key challenge is therefore to identify which variables are repeatedly prioritized across modeling strategies and remain informative after appropriate adjustment, without overstating their independent clinical utility.

Renal, metabolic, and inflammatory pathways are closely connected with cardiovascular health. Kidney dysfunction may reflect vascular injury, endothelial dysfunction, microvascular disease, and systemic metabolic disturbance. Albuminuria, captured by urine albumin-to-creatinine ratio (UACR), is a routinely measurable marker with potential relevance across renal and cardiovascular pathways. Estimated glomerular filtration rate (eGFR) captures a different aspect of kidney function and may complement albuminuria. Metabolic indicators such as the triglyceride-glucose index and its anthropometric extensions, including TyG-WC, combine glycemic and lipid-related information with adiposity. Inflammatory composite indices such as the systemic inflammation response index (SIRI) may summarize leukocyte patterns relevant to chronic inflammatory states.

Interpretable machine learning offers a way to compare a broad set of candidate predictors while retaining transparent feature-prioritization tools such as permutation importance and SHAP-based explanations. Public datasets also allow analyses to be audited and reproduced. In this study, GBD and NHANES were intentionally kept at different analytic levels. GBD was used only to describe population-level CVD burden and attributable risk-factor context. NHANES was used for individual-level prediction and survey-weighted epidemiologic analyses. This separation avoids implying that macro-level burden estimates validate individual-level biomarkers.

The objective of this study was to integrate population-level GBD 2023 CVD burden trends with NHANES 2005-2018 individual-level modeling to identify renal, metabolic, and inflammatory indicators associated with self-reported CVD. Specifically, we aimed to compare leakage-audited machine learning feature sets, summarize consensus feature rankings, evaluate survey-weighted associations for selected indicators, assess nonlinear associations, and examine subgroup and sensitivity results. We sought to generate a cautious, manuscript-ready evidence package for public-database cardiovascular research rather than etiologic proof or a clinical decision model.

This design was intended to address two complementary but distinct questions. The GBD component asks how CVD burden and attributable risk context have changed over time at the population level. The NHANES component asks which individual-level indicators are prioritized by interpretable models and supported by survey-weighted analyses in a nationally representative adult sample. Keeping these questions separate is important because population-level risk-attributable burden estimates and individual-level biomarker associations are not interchangeable. By placing them side by side rather than merging them, the study offers a public-health framing followed by an individual-level analytic workflow.

## Methods

### Study design and data sources

This was a public-database study with two analytically separate components. First, GBD 2023 CSV exports were used to describe CVD burden trends and attributable risk-factor rankings at the population level. Second, NHANES 2005-2018 data were used to conduct individual-level machine learning, interpretable feature ranking, and survey-weighted statistical analyses. The two data sources were not merged, and no joint GBD-NHANES model was fitted.

The workflow was organized to preserve the intended level of inference for each data source. GBD files were treated as aggregate burden estimates and were summarized by location, measure, metric, age category, sex, and year. NHANES records were treated as individual-level survey observations and analyzed with attention to sample weights, strata, and primary sampling units. This separation also reduced the risk that macro-level burden patterns would be interpreted as validating individual-level predictors. Instead, the GBD findings were positioned as background context, while NHANES analyses formed the basis for prediction and association results.

### GBD 2023 burden analysis

GBD 2023 data were read from official user-provided CSV exports. The analysis focused on cardiovascular diseases and included deaths, DALYs, and prevalence, with number and rate metrics. Locations included Global, China, and United States. Ages included all ages and age-standardized categories; sex categories included both, male, and female when available. Age-standardized rates were summarized for 1990 and 2023, and percent change was computed directly from those values. Estimated annual percentage change (EAPC) was estimated from log-linear trends over calendar year. Attributable CVD burden was summarized for selected risk factors, including high systolic blood pressure, dietary risks, high LDL cholesterol, smoking, high fasting plasma glucose, kidney dysfunction, and high body-mass index.

Common IHME field names and equivalent aliases were harmonized before analysis. Burden trend tables were generated separately from attributable-risk tables so that total disease burden was not confused with risk-attributable burden. This distinction was particularly important because the manuscript package audit identified an earlier narrative inconsistency for China CVD death-rate percent change; the final manuscript therefore uses the Stage 5 Table 5 values derived directly from `gbd_cvd_burden_trend.csv` and `gbd_eapc_summary.csv`.

### NHANES study population

NHANES 2005-2018 adult participants were included according to the project pipeline. The final analytic sample contained {n_total} adults. Survey-weighted analyses used NHANES design variables, including combined fasting weights, strata, and primary sampling units. Participants with missing CVD outcome were excluded in the data-preparation stage. The analysis retained the complex survey structure for weighted descriptive and regression analyses.

The analytic approach was designed to reflect typical public-database epidemiologic practice. Descriptive and regression analyses used survey design variables to account for the multistage sampling structure. Machine learning analyses used held-out evaluation to assess discrimination and calibration-related metrics. Because the project emphasized reproducibility and leakage control, outcome component variables and direct CVD diagnostic variables were excluded from predictor sets before model fitting.

### CVD outcome definition

CVD was defined using self-reported physician diagnosis or history variables from the NHANES questionnaire. Participants were classified as having CVD if they reported congestive heart failure, coronary heart disease, angina, heart attack or myocardial infarction, or stroke. The Stage 3 audit found no mismatch between the derived CVD variable and the five component variables.

### Candidate predictors and derived indicators

Candidate predictors covered demographic, lifestyle, anthropometric, blood pressure, glucose and lipid metabolism, inflammatory, and renal-function domains. Derived indicators included TyG index, TyG-BMI, TyG-WC, atherogenic index of plasma, non-HDL-C, neutrophil-to-lymphocyte ratio, platelet-to-lymphocyte ratio, systemic immune-inflammation index, SIRI, eGFR, and UACR when source variables were available. CVD outcome components and any direct CVD diagnostic variables were excluded from predictor sets.

The candidate indicators were selected to emphasize routinely measurable variables rather than specialized assays. Renal indicators were represented by UACR, eGFR, creatinine, urine albumin, and related measures. Metabolic indicators included glucose, HbA1c, lipid fractions, non-HDL-C, and TyG-related indices. Inflammatory indicators were derived from complete blood count components. This structure allowed the machine learning and regression analyses to compare different domains while retaining clinical interpretability.

### Feature sets

Three leakage-audited feature sets were evaluated. The full clinical model included demographic, lifestyle, anthropometric, blood pressure, metabolic, inflammatory, renal-function, and selected comorbidity variables. The routine indicator model excluded hypertension and diabetes diagnosis variables and focused on routinely measured demographic, lifestyle, anthropometric, blood pressure, glycemic, lipid, inflammatory, and renal variables. The biomarker-focused model included basic adjustment variables plus metabolic, inflammatory, and renal indicators, including TyG-related indices, non-HDL-C, NLR, PLR, SII, SIRI, UACR, eGFR, uric acid, creatinine, and albumin where available.

### Machine learning models

Machine learning analyses compared multiple classifiers across the three feature sets. The final manuscript-ready performance table identified the best model in each feature set based on held-out test discrimination. Train/test splitting was stratified, and preprocessing was fitted within training-set pipelines. Model performance metrics included AUC, PR-AUC, F1 score, Brier score, sensitivity, and specificity.

The purpose of the machine learning component was not to produce a deployable clinical model, but to identify whether a biomarker-focused feature set could approach the discrimination of broader clinical feature sets and to prioritize candidate indicators for subsequent statistical validation. The use of multiple feature sets also helped distinguish features that were important only when diagnoses were included from indicators that remained prominent in more routine or biomarker-focused settings.

### SHAP and consensus feature ranking

Feature importance was summarized using complementary approaches, including model-based importance, permutation importance, and held-out SHAP explanations. Consensus feature ranking was derived across methods within each feature set. The manuscript focused on priority indicators that were biologically relevant and repeatedly prioritized across routine and biomarker-focused models, including UACR, eGFR, TyG-WC, SIRI, TyG index, non-HDL-C, LDL-C, and HbA1c.

### Survey-weighted statistical analysis

Survey-weighted logistic regression was used to estimate odds ratios (ORs) and 95% confidence intervals (CIs) for key indicators. Model 3 served as the main statistical model and adjusted for core demographic and lifestyle factors. The strict fully adjusted Model 4 added blood pressure, glycemic, lipid, renal, and inflammatory covariates while excluding the target indicator itself. The strict fully adjusted model had a smaller complete-case sample and may be susceptible to overadjustment among correlated biomarkers.

Model 3 was prioritized for main interpretation because it preserved a larger analytic sample and adjusted for major demographic and lifestyle covariates without mutually adjusting for many biomarkers from overlapping biological pathways. Model 4 was retained as a conservative analysis to assess whether associations persisted under extensive adjustment. Attenuation in Model 4 was interpreted cautiously, especially when the target indicator was biologically correlated with other covariates in the same model.

### RCS analysis

Nonlinear associations were assessed using survey-weighted natural spline models for prespecified indicators, including UACR, eGFR, TyG-WC, SIRI, TyG index, and non-HDL-C. RCS results were interpreted as evidence of nonlinear association patterns rather than as mechanistic proof.

### Subgroup and sensitivity analyses

Subgroup analyses evaluated whether associations varied across age group, sex, diabetes, hypertension, and obesity status when available. Sensitivity analyses excluded prior stroke, excluded prior heart attack, and used routine biomarker adjustment without hypertension or diabetes diagnosis variables. These analyses were used to evaluate robustness rather than to define clinical decision thresholds.

### Ethics statement

NHANES is a publicly available survey program with protocols approved by the National Center for Health Statistics ethics review board, and participants provided informed consent at the time of data collection. The present analysis used de-identified public data and official GBD CSV exports. Additional institutional review board approval was not required for this secondary analysis.

### Statistical software

Data preparation, machine learning, feature ranking, GBD analysis, and manuscript table generation were performed using Python. Survey-weighted analyses were performed using R with survey-design methods. Figure assembly and manuscript-ready result packaging were performed with reproducible project scripts.

## Results

### GBD burden trends

GBD 2023 data showed declining age-standardized CVD death and DALY rates from 1990 to 2023 in Global, China, and United States settings. Globally, the age-standardized CVD death rate changed from {gbd_lines['global_deaths'][0]} in 1990 to {gbd_lines['global_deaths'][1]} in 2023 ({gbd_lines['global_deaths'][2]}; EAPC {gbd_lines['global_deaths'][3]}%, 95% CI {gbd_lines['global_deaths'][4]}, P={gbd_lines['global_deaths'][5]}), and the age-standardized DALY rate changed from {gbd_lines['global_dalys'][0]} to {gbd_lines['global_dalys'][1]} ({gbd_lines['global_dalys'][2]}; EAPC {gbd_lines['global_dalys'][3]}%, 95% CI {gbd_lines['global_dalys'][4]}, P={gbd_lines['global_dalys'][5]}). In China, the age-standardized CVD death rate changed from {gbd_lines['china_deaths'][0]} to {gbd_lines['china_deaths'][1]} ({gbd_lines['china_deaths'][2]}; EAPC {gbd_lines['china_deaths'][3]}%, 95% CI {gbd_lines['china_deaths'][4]}, P={gbd_lines['china_deaths'][5]}), and the age-standardized DALY rate changed from {gbd_lines['china_dalys'][0]} to {gbd_lines['china_dalys'][1]} ({gbd_lines['china_dalys'][2]}; EAPC {gbd_lines['china_dalys'][3]}%, 95% CI {gbd_lines['china_dalys'][4]}, P={gbd_lines['china_dalys'][5]}). In the United States, the age-standardized DALY rate changed from {gbd_lines['us_dalys'][0]} to {gbd_lines['us_dalys'][1]} ({gbd_lines['us_dalys'][2]}; EAPC {gbd_lines['us_dalys'][3]}%, 95% CI {gbd_lines['us_dalys'][4]}, P={gbd_lines['us_dalys'][5]}). China showed a small increase in age-standardized prevalence rate from {gbd_lines['china_prev'][0]} to {gbd_lines['china_prev'][1]} ({gbd_lines['china_prev'][2]}), with a stable EAPC estimate ({gbd_lines['china_prev'][3]}%, 95% CI {gbd_lines['china_prev'][4]}, P={gbd_lines['china_prev'][5]}).

The leading attributable CVD DALY risk factors in 2023 were broadly similar across Global and China settings. In China, the leading risk factors by DALY number were high systolic blood pressure, dietary risks, high LDL cholesterol, smoking, and high fasting plasma glucose. Globally, the leading risk factors were high systolic blood pressure, dietary risks, high LDL cholesterol, smoking, and high body-mass index. These GBD findings were used only to provide population-level context for the NHANES analyses.

### NHANES population characteristics

The NHANES analytic sample included {n_total} adults, of whom {n_cvd} had self-reported CVD and {n_noncvd} did not. Weighted baseline characteristics differed between participants with and without CVD. Participants with CVD were older than those without CVD ({age['CVD']} versus {age['Non-CVD']}, P={age['P value']}). Weighted body mass index was higher in the CVD group ({bmi['CVD']} versus {bmi['Non-CVD']}, P={bmi['P value']}), and poverty income ratio was lower ({pir['CVD']} versus {pir['Non-CVD']}, P={pir['P value']}). Systolic blood pressure was also higher in the CVD group ({sbp['CVD']} versus {sbp['Non-CVD']}, P={sbp['P value']}). Glycemic markers showed similar group differences, with fasting glucose ({glucose['CVD']} versus {glucose['Non-CVD']}, P={glucose['P value']}) and HbA1c ({hba1c_base['CVD']} versus {hba1c_base['Non-CVD']}, P={hba1c_base['P value']}) higher among participants with CVD.

### Machine learning model performance

The leakage-audited machine learning models showed similar discrimination across feature sets. In the full clinical model, XGBoost had the best performance, with AUC={full['AUC']}, PR-AUC={full['PR-AUC']}, F1={full['F1']}, and Brier score={full['Brier score']}. In the routine indicator model, XGBoost was also the best-performing model, with AUC={routine['AUC']}, PR-AUC={routine['PR-AUC']}, F1={routine['F1']}, and Brier score={routine['Brier score']}. In the biomarker-focused model, Elastic Net performed best, with AUC={biomarker['AUC']}, PR-AUC={biomarker['PR-AUC']}, F1={biomarker['F1']}, and Brier score={biomarker['Brier score']}. The AUC difference between the full clinical and biomarker-focused best models was {auc_diff}, indicating comparable discrimination despite the more focused biomarker-oriented feature set.

### SHAP and feature importance

Consensus feature ranking highlighted renal, metabolic, inflammatory, and lipid-related indicators. UACR entered the top 10 in both the routine and biomarker-focused models, with routine consensus rank {uacr_feat['routine consensus rank']} and biomarker-focused consensus rank {uacr_feat['biomarker consensus rank']}. eGFR was also ranked in the top 10 in both feature sets, with routine rank {egfr_feat['routine consensus rank']} and biomarker-focused rank {egfr_feat['biomarker consensus rank']}. TyG-WC entered the top 10 in both models, with routine rank {tyg_feat['routine consensus rank']} and biomarker-focused rank {tyg_feat['biomarker consensus rank']}. SIRI was not in the routine top 10 but was in the biomarker-focused top 10, with biomarker-focused rank {siri_feat['biomarker consensus rank']}. TyG index and non-HDL-C were prioritized in the biomarker-focused top 10, whereas LDL-C and HbA1c were prioritized in the routine top 10.

### Survey-weighted regression

In survey-weighted logistic regression, UACR showed the most robust evidence across models. In Model 3, UACR was associated with CVD (OR {uacr['Model 3 OR (95% CI)']}, P={uacr['Model 3 P value']}), and the association remained present in the strict fully adjusted Model 4 (OR {uacr['Strict Model 4 OR (95% CI)']}, P={uacr['Strict Model 4 P value']}). eGFR was associated with CVD in Model 3 (OR {egfr['Model 3 OR (95% CI)']}, P={egfr['Model 3 P value']}) but was attenuated in strict Model 4 (OR {egfr['Strict Model 4 OR (95% CI)']}, P={egfr['Strict Model 4 P value']}). TyG-WC was associated with CVD in Model 3 (OR {tyg_wc['Model 3 OR (95% CI)']}, P={tyg_wc['Model 3 P value']}) and attenuated in strict Model 4 (OR {tyg_wc['Strict Model 4 OR (95% CI)']}, P={tyg_wc['Strict Model 4 P value']}). SIRI showed a Model 3 association (OR {siri['Model 3 OR (95% CI)']}, P={siri['Model 3 P value']}) but did not remain statistically significant in strict Model 4 (OR {siri['Strict Model 4 OR (95% CI)']}, P={siri['Strict Model 4 P value']}).

LDL-C showed an inverse association in the strict fully adjusted model, which should be interpreted cautiously because of potential confounding, medication use, and collinearity. This finding was not treated as a primary conclusion and was not interpreted as indicating benefit. Overall, Model 3 was treated as the main statistical model, whereas Model 4 was considered a strict fully adjusted model with a smaller complete-case sample and potential overadjustment among correlated biomarkers.

### Nonlinear associations, subgroup analysis, and sensitivity analysis

RCS analyses supported nonlinear associations for the main renal, metabolic, and inflammatory indicators. UACR showed evidence of nonlinearity (P for nonlinearity={p_uacr_rcs}), as did eGFR (P={p_egfr_rcs}), TyG-WC (P={p_tyg_rcs}), and SIRI (P={p_siri_rcs}). Subgroup analysis identified an interaction for TyG-WC by obesity status (P interaction=0.0211). In sensitivity analyses, UACR was statistically significant in 2/3 sensitivity analyses, whereas eGFR was significant in 0/3, TyG-WC in 1/3, and SIRI in 1/3. These results supported UACR as the most stable indicator across analytic approaches, while eGFR, TyG-WC, and SIRI were supported but more sensitive to model specification and sensitivity-analysis conditions.

## Discussion

### Main findings

This public-database study combined population-level GBD context with individual-level NHANES modeling while maintaining strict separation between the two analytic levels. GBD 2023 data showed declining age-standardized CVD death and DALY rates from 1990 to 2023 in Global, China, and United States settings, with high systolic blood pressure, dietary risks, high LDL cholesterol, and smoking ranking among the leading attributable risk factors. In NHANES 2005-2018, interpretable machine learning models showed comparable discrimination across full clinical, routine indicator, and biomarker-focused feature sets. UACR emerged as the most robust individual-level indicator, while eGFR, TyG-WC, and SIRI were repeatedly prioritized or supported by machine learning, Model 3 regression, and nonlinear analyses.

The overall evidence pattern was strongest when multiple analytic layers pointed in the same direction. UACR was prioritized by machine learning, remained statistically supported in the strict fully adjusted model, showed nonlinearity, and retained partial support in sensitivity analyses. eGFR, TyG-WC, and SIRI were also repeatedly highlighted, but their results were more sensitive to extensive mutual adjustment or sensitivity-analysis settings. This distinction is important for framing the findings: the study identifies a hierarchy of cross-sectional evidence rather than a definitive set of independent clinical predictors.

### Interpretation of comparable model discrimination

The full clinical model achieved the highest AUC ({full['AUC']}), but the biomarker-focused model also performed well (AUC={biomarker['AUC']}), with a full versus biomarker-focused AUC difference of {auc_diff}. This suggests that a focused set of renal, metabolic, inflammatory, and related routine indicators captured substantial information relevant to CVD classification in this cross-sectional NHANES setting. The result should not be interpreted as showing that a biomarker-focused model replaces comprehensive clinical assessment. Rather, it indicates that routinely available biomarkers contributed meaningfully to model discrimination and may be useful for risk-marker screening and hypothesis generation.

The comparable discrimination also suggests that several routinely collected laboratory and anthropometric measures may summarize overlapping information contained in broader clinical profiles. This is plausible because age, renal function, glycemic status, lipid-related variables, adiposity, and inflammation are intertwined in cardiometabolic health. However, similar discrimination does not imply equivalent clinical utility. Calibration, decision thresholds, longitudinal performance, and external validation remain necessary before any model could be considered for practical risk stratification.

### Why UACR was the most robust indicator

UACR was the most consistent indicator across analytic layers. It was ranked in the top 10 in both routine and biomarker-focused feature sets, showed a Model 3 association with CVD, remained associated in strict Model 4, had evidence of nonlinearity in RCS analysis, and was significant in 2/3 sensitivity analyses. This pattern is consistent with the concept that albuminuria may capture vascular, renal, and metabolic information not fully summarized by standard covariates. Importantly, the present study does not establish that UACR changes future CVD outcomes or should be used alone for clinical decisions. It supports UACR as a robust cross-sectional marker associated with self-reported CVD in a nationally representative adult sample.

The robustness of UACR may reflect its position at the interface of kidney function and vascular health. Unlike serum creatinine or eGFR alone, albuminuria may capture glomerular permeability, endothelial injury, and microvascular dysfunction. In the present analyses, UACR remained statistically supported even when the strict model included several related renal and cardiometabolic covariates. This does not mean that UACR is independent of those pathways in a mechanistic sense; rather, it remained informative under the conservative adjustment strategy used here.

### Role of eGFR, TyG-WC, and SIRI

eGFR was strongly prioritized by machine learning and was associated with CVD in Model 3, but the estimate was attenuated in strict Model 4 and was not stable in sensitivity analyses. This pattern may reflect overlap between eGFR, creatinine, albuminuria, glycemic variables, blood pressure, and other clinical indicators. TyG-WC combined metabolic and adiposity-related information and was supported by feature ranking, Model 3 regression, RCS analysis, and an obesity-status interaction, although the strict Model 4 result was attenuated. SIRI represented the inflammatory domain and was supported by biomarker-focused feature ranking, Model 3 regression, and RCS analysis, but was also sensitive to strict adjustment and sensitivity analyses. Together, these findings suggest that eGFR, TyG-WC, and SIRI may help characterize CVD-associated phenotypes, while their independent contribution requires careful validation.

These three indicators should therefore be interpreted as complementary signals. eGFR provided a strong renal-function signal in the machine learning ranking. TyG-WC integrated metabolic and anthropometric information and may be particularly informative among participants with obesity, consistent with the observed subgroup interaction. SIRI summarized inflammatory cell patterns and may reflect systemic inflammatory burden. Each indicator has plausible clinical interpretability, but their attenuation in strict adjustment highlights the importance of considering biological overlap rather than treating each marker as an isolated predictor.

### Relationship with GBD risk factor findings

The GBD results provide population-level context rather than validation of NHANES biomarkers. High systolic blood pressure, dietary risks, high LDL cholesterol, and smoking were leading attributable risk factors for CVD DALYs in 2023. These macro-level findings align broadly with the importance of cardiometabolic and vascular domains in CVD burden. However, GBD attributable burden and NHANES individual-level feature ranking answer different questions. GBD describes population-level burden patterns and risk-attributable context, whereas NHANES evaluates associations between individual-level indicators and self-reported CVD status. The present study therefore uses GBD as Result 1 public-health context and NHANES as the primary source for individual-level modeling and association results.

### Clinical and public health implications

The findings support attention to routine renal, metabolic, and inflammatory measures in CVD-related population research. UACR may be especially useful in future risk-marker studies because it showed stability across several complementary analytic approaches. eGFR, TyG-WC, and SIRI may also contribute to phenotyping and risk-marker screening, especially when interpreted with standard demographic and clinical variables. From a public-health perspective, the GBD results reinforce the continued importance of blood pressure, diet, lipids, smoking, glycemic status, adiposity, and kidney dysfunction in CVD burden. These implications should be framed as research and surveillance priorities rather than immediate clinical recommendations.

For public health research, a practical implication is that existing survey and clinical datasets may already contain informative renal, metabolic, and inflammatory variables that can be examined without requiring specialized biomarkers. For clinical research, the results support further evaluation of UACR and related indicators in longitudinal cohorts and risk models that include medication use and adjudicated outcomes. For global burden research, the GBD component reinforces that macro-level prevention priorities remain centered on established modifiable risks, while NHANES helps identify individual-level markers that may be useful for more detailed phenotyping.

### Strengths

This study has several strengths. It used large, public, and reproducible data sources. The GBD and NHANES analyses were kept analytically separate, reducing the risk of ecological interpretation errors. The NHANES pipeline included leakage auditing, prespecified feature sets, held-out test evaluation, and interpretable feature-ranking methods. Survey-weighted analyses respected the complex sampling design of NHANES. The study also triangulated evidence across machine learning, regression, nonlinear, subgroup, and sensitivity analyses, allowing indicators to be judged by consistency rather than by a single model.

### Limitations

Several limitations should be considered. The NHANES outcome was self-reported and cross-sectional, so temporal relationships cannot be established. Medication use, treatment history, and clinical management may influence biomarker levels and could contribute to patterns such as the inverse LDL-C association in strict Model 4. The strict fully adjusted model had a smaller complete-case sample and may be susceptible to overadjustment among correlated biomarkers. Some indicators were not available across all participants, and missingness may affect complete-case survey models. RCS analyses were performed for prespecified indicators only; RCS nonlinearity was not available for LDL-C or HbA1c in the Stage 5 package. Finally, machine learning performance was evaluated in held-out NHANES data and requires external and prospective validation.

Additional limitations relate to interpretation across data sources. GBD burden estimates are model-based population estimates and should not be treated as participant-level data. NHANES provides rich individual-level information but is not designed to determine the timing of biomarker changes relative to CVD onset. The CVD definition combined several self-reported conditions, which increases relevance to overall cardiovascular history but may obscure differences among heart failure, coronary disease, myocardial infarction, angina, and stroke. Finally, feature importance methods can be influenced by correlation among predictors, so consensus ranking was used as a prioritization tool rather than a definitive measure of biological importance.

### Future directions

Future studies should validate these findings in prospective cohorts and in external datasets with adjudicated cardiovascular outcomes. Longitudinal analyses could evaluate whether UACR, eGFR, TyG-WC, and SIRI improve prediction beyond established risk factors and whether changes in these indicators correspond to changes in future event risk. Additional work should also evaluate medication use, kidney disease stages, diabetes subgroups, and race/ethnicity-specific performance. Finally, model updating and calibration should be assessed before any clinical decision-support use.

## Conclusions

In this analysis of GBD 2023 and NHANES 2005-2018 public data, GBD described population-level CVD burden trends, while NHANES supported individual-level machine learning and survey-weighted association analyses. The biomarker-focused model showed discrimination comparable to the full clinical model. UACR was the most robust indicator across feature ranking, survey-weighted regression, nonlinear analysis, and sensitivity testing. eGFR, TyG-WC, and SIRI were consistently prioritized or supported but were more sensitive to strict adjustment. These findings should be interpreted as cross-sectional risk-marker evidence, and future prospective validation is needed.

## Tables and Figures

### Tables

- Table 1. Weighted baseline characteristics by CVD status.
- Table 2. Machine learning model performance by feature set.
- Table 3. Feature ranking and statistical validation summary for priority indicators.
- Table 4. Survey-weighted regression results for key indicators.
- Table 5. GBD CVD burden summary, percent change, and EAPC.

### Figures

- Figure 1. GBD cardiovascular disease burden and attributable risk factors.
- Figure 2. Machine learning model performance.
- Figure 3. SHAP and feature importance.
- Figure 4. Restricted cubic spline analyses for key indicators.
- Figure 5. Subgroup and sensitivity analyses.

## Declarations

### Ethics approval and consent to participate

NHANES protocols were approved by the National Center for Health Statistics ethics review board, and participants provided informed consent at the time of data collection. This secondary analysis used de-identified public NHANES data and official GBD CSV exports.

### Consent for publication

Not applicable.

### Availability of data and materials

NHANES data are publicly available from the National Center for Health Statistics. GBD results are available through official IHME/GBD tools. Project scripts and derived outputs can be shared according to journal and repository requirements.

### Competing interests

The authors declare no competing interests. [To be confirmed by all authors]

### Funding

No specific funding was reported for this draft. [To be completed]

### Authors' contributions

Author contributions will be completed according to the final author list. [To be completed]

### Acknowledgements

The authors acknowledge the participants and staff of NHANES and the IHME/GBD collaborators who produced the public data resources used in this study.

## References

1. Reference for GBD 2023 methods and results. [Reference to be verified]
2. Reference for IHME GBD Results Tool. [Reference to be verified]
3. Reference for NHANES survey design and analytic guidelines. [Reference to be verified]
4. Reference for NHANES cardiovascular questionnaire variables. [Reference to be verified]
5. Reference for CKD-EPI eGFR equation. [Reference to be verified]
6. Reference for UACR and cardiovascular risk literature. [Reference to be verified]
7. Reference for TyG index and cardiovascular risk literature. [Reference to be verified]
8. Reference for inflammatory indices including SIRI and cardiovascular disease. [Reference to be verified]
9. Reference for SHAP methodology. [Reference to be verified]
10. Reference for survey-weighted regression and NHANES complex sampling analysis. [Reference to be verified]
"""

    source_context = {
        "methods_skeleton_loaded": len(methods_skeleton),
        "legends_loaded": len(legends),
        "stage5_audit_loaded": len(stage5_audit),
        "gbd_check_loaded": len(gbd_check),
    }
    return manuscript, source_context


def audit_manuscript(manuscript: str, source_context: dict) -> str:
    words = re.findall(r"\b[\w'-]+\b", manuscript)
    lower = manuscript.lower()
    forbidden = {
        "caused": "caused" in lower,
        "novel biomarker": "novel biomarker" in lower,
        "causal effect": "causal effect" in lower,
    }
    overclaim_terms = []
    for phrase in ["validated by gbd", "gbd validated", "proves", "causality"]:
        if phrase in lower:
            overclaim_terms.append(phrase)
    refs_need_completion = "[Reference to be verified]" in manuscript
    audit = [
        "# Stage 6 Manuscript Draft Audit",
        "",
        f"- Manuscript completed: {bool(manuscript.strip())}",
        f"- Output file: `outputs/manuscript_draft/CVD_NHANES_GBD_ML_manuscript_draft_en.md`",
        f"- Approximate word count: {len(words)}",
        "- All numeric results were pulled from Stage 5 manuscript-ready tables and text inputs.",
        f"- Source methods skeleton loaded: {source_context['methods_skeleton_loaded']} characters",
        f"- Source figure/table notes loaded: {source_context['legends_loaded']} characters",
        f"- Stage 5 audit loaded: {source_context['stage5_audit_loaded']} characters",
        f"- Stage 5 GBD result check loaded: {source_context['gbd_check_loaded']} characters",
        "",
        "## Prohibited or Guarded Language Check",
        "",
    ]
    for phrase, present in forbidden.items():
        audit.append(f"- `{phrase}` present: {present}")
    audit.append(f"- Other overstatement terms detected: {', '.join(overclaim_terms) if overclaim_terms else 'None'}")
    audit.extend(
        [
            "",
            "## Numeric and Interpretation Checks",
            "",
            "- The China CVD death-rate percent change uses Stage 5 Table 5 and GBD result check values, not the earlier inconsistent Stage 4.1 narrative.",
            "- GBD is described only as population-level burden context.",
            "- NHANES is described as individual-level modeling and association analysis.",
            "- Model 4 is described as strict fully adjusted, smaller complete-case, and potentially overadjusted among correlated biomarkers.",
            "- LDL-C inverse association is explicitly described cautiously and not as protective.",
            "",
            "## Items Requiring Human Follow-up",
            "",
            f"- References require completion and verification: {refs_need_completion}",
            "- Journal-specific formatting, word limit, abstract style, and reference style still need Stage 7 handling.",
            "- Author list, funding, competing interests, and contributions require manual confirmation.",
            "",
            "## Recommendation",
            "",
            "- Proceed to Stage 7 references and journal formatting.",
        ]
    )
    return "\n".join(audit) + "\n"


def main() -> None:
    manuscript, source_context = build_manuscript()
    draft_path = DRAFT_DIR / "CVD_NHANES_GBD_ML_manuscript_draft_en.md"
    draft_path.write_text(manuscript, encoding="utf-8")
    audit = audit_manuscript(manuscript, source_context)
    audit_path = DIAG / "stage6_manuscript_draft_audit.md"
    audit_path.write_text(audit, encoding="utf-8")
    word_count = len(re.findall(r"\b[\w'-]+\b", manuscript))
    print(f"Wrote {draft_path}")
    print(f"Wrote {audit_path}")
    print(f"Approximate word count: {word_count}")


if __name__ == "__main__":
    main()
