# GBD + NHANES CVD Burden and Interpretable ML Project
This repository contains analysis code for the manuscript: "Interpretable machine learning identifies renal, metabolic and inflammatory indicators associated with cardiovascular disease: evidence from NHANES and GBD."
This project scaffolds a reproducible public-database study:

**Cardiovascular disease burden trends using GBD and metabolic-inflammatory-renal indicator screening using NHANES with interpretable machine learning and weighted epidemiology.**

GBD and NHANES are intentionally analyzed separately. GBD is used for macro-level burden trends; NHANES is used for individual-level CVD prediction, feature screening, and survey-weighted validation.

## 1. Environment

From PowerShell:

```powershell
cd C:\Users\Administrator\cvd_gbd_nhanes_ml
conda env create -f environment.yml
conda activate cvd-gbd-nhanes
```

If you prefer pip-only for Python:

```powershell
pip install -r requirements.txt
```

The R survey script also needs R packages listed in `environment.yml`: `survey`, `optparse`, `readr`, `dplyr`, and `ggplot2`.

## 2. Run Everything

Default NHANES range is 2005-2018:

```powershell
python scripts/run_all.py --start-year 2005 --end-year 2018 --download
```

For 1999-2018:

```powershell
python scripts/run_all.py --start-year 1999 --end-year 2018 --download
```

For a TyG-focused fasting-subsample run:

```powershell
python scripts/run_all.py --start-year 2005 --end-year 2018 --download --require-fasting-core
```

If no GBD CSV files are present in `data/raw/gbd`, the GBD step is skipped. No placeholder or fabricated results are written.

## 3. Step-by-Step Workflow

Prepare NHANES analytic data:

```powershell
python scripts/nhanes_prepare.py --start-year 2005 --end-year 2018 --download --impute-method iterative
```

For a TyG-centered fasting-subsample analysis, add:

```powershell
python scripts/nhanes_prepare.py --start-year 2005 --end-year 2018 --download --require-fasting-core
```

Train and evaluate machine learning models:

```powershell
python scripts/ml_modeling.py
```

Run feature selection and SHAP:

```powershell
python scripts/feature_selection.py
```

Run NHANES survey-weighted epidemiologic validation:

```powershell
Rscript R/weighted_survey_analysis.R --project-root .
```

Run GBD analysis after placing GBD CSV files in `data/raw/gbd`:

```powershell
python scripts/gbd_analysis.py
```

Generate a short manuscript-style summary from completed outputs:

```powershell
python scripts/summarize_results.py
```

## 4. NHANES Data Handling

`scripts/nhanes_prepare.py` downloads or reads XPT files for:

`DEMO`, `MCQ`, `BMX`, `BPX`, `GLU`, `GHB`, `TCHOL`, `HDL`, `TRIGLY`, `CBC`, `CRP/HSCRP`, `BIOPRO`, `ALB_CR`, `SMQ`, `ALQ`, `PAQ`, `SLQ`, `BPQ`, and `DIQ`.

CVD is defined as any self-reported congestive heart failure, coronary heart disease, angina, heart attack, or stroke.

Derived indicators include TyG, TyG-BMI, TyG-WC, AIP, non-HDL-C, remnant cholesterol when LDL-C is available, NLR, PLR, SII, SIRI, CKD-EPI 2021 eGFR, and UACR.

The script applies:

- age >= 20 years
- pregnancy exclusion when `RIDEXPRG` is available
- non-missing CVD outcome
- optional fasting-core restriction for TyG-family analyses
- variable removal when missingness is >30%
- continuous-variable winsorization with a saved log
- IterativeImputer/MICE-style numeric imputation and most-frequent categorical imputation
- multi-cycle NHANES weights: 2-year weights divided by the number of included cycles

Missing modules or variables are logged as warnings and skipped.

## 5. Optional Linked Mortality

If you have NHANES linked mortality files converted to CSV, place them in `data/raw/nhanes_mortality` and run:

```powershell
python scripts/merge_mortality.py
```

This derives all-cause mortality from `MORTSTAT == 1` and cardiovascular mortality from `UCOD_LEADING` categories 1 or 5. Mortality analysis is optional and is not fabricated when files are absent.

## 6. GBD Input

Place user-exported GBD CSV files in `data/raw/gbd`. Accepted columns include:

`location/location_name`, `year/year_id`, `sex/sex_name`, `age/age_name`, `measure/measure_name`, `metric/metric_name`, `cause/cause_name`, `rei/rei_name`, and `value/val`.

The GBD script outputs CVD deaths, DALYs, prevalence, age-standardized rates, EAPC estimates, risk-factor ranking, and age-sex heatmaps.

## 7. Expected Outputs

Main tables:

- `outputs/tables/table1_baseline.csv`
- `outputs/tables/table2_model_performance.csv`
- `outputs/tables/table3_feature_ranking.csv`
- `outputs/tables/table4_weighted_regression.csv`

Main figures:

- `outputs/figures/fig1_gbd_trend.png`
- `outputs/figures/fig2_roc_pr_calibration.png`
- `outputs/figures/fig3_shap_summary.png`
- `outputs/figures/fig4_rcs.png`
- `outputs/figures/fig5_subgroup_forest.png`

Other useful outputs:

- `outputs/tables/inclusion_exclusion_flow.csv`
- `outputs/tables/missingness_report.csv`
- `outputs/tables/winsorization_log.csv`
- `outputs/tables/imputation_log.csv`
- `outputs/tables/feature_importance_long.csv`
- `outputs/tables/gbd_eapc.csv`
- `outputs/manuscript_results/results_summary.md`
- detailed logs in `outputs/logs`

## 8. Notes for Manuscript Use

Use the generated outputs as analysis products only after reviewing variable availability, missingness, model diagnostics, and survey design assumptions. The pipeline does not force GBD and NHANES into one model, and it never writes simulated study results.
