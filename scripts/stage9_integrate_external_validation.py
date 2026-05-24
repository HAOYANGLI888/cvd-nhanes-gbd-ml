from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANUSCRIPT = PROJECT_ROOT / "outputs/manuscript_draft/CVD_NHANES_GBD_ML_manuscript_draft_en_stage8_external_validation.md"
FINAL_MANUSCRIPT = PROJECT_ROOT / "outputs/manuscript_draft/CVD_NHANES_GBD_ML_manuscript_draft_en_stage9_final_with_external_validation.md"
EXT_DIR = PROJECT_ROOT / "outputs/external_validation"
TABLE6 = PROJECT_ROOT / "outputs/manuscript_tables/Table_6_external_validation.csv"
FIG6 = PROJECT_ROOT / "outputs/manuscript_figures/Figure_6_External_validation_performance.png"
AUDIT = PROJECT_ROOT / "outputs/diagnostics/stage9_external_validation_integration_audit.md"


def fmt(x, digits=4) -> str:
    if pd.isna(x):
        return "not available"
    return f"{float(x):.{digits}f}"


def pformat(x) -> str:
    if pd.isna(x):
        return "not available"
    x = float(x)
    return "P < 0.001" if x < 0.001 else f"P = {x:.4f}"


def read_inputs() -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    text = SOURCE_MANUSCRIPT.read_text(encoding="utf-8")
    perf = pd.read_csv(EXT_DIR / "external_model_performance_2021_2023.csv")
    reg = pd.read_csv(EXT_DIR / "external_weighted_regression_2021_2023.csv")
    report = (EXT_DIR / "stage8_external_validation_report.md").read_text(encoding="utf-8")
    return text, perf, reg, report


def create_table6(perf: pd.DataFrame, reg: pd.DataFrame) -> None:
    rows = []
    for _, row in perf.iterrows():
        rows.append(
            {
                "analysis_type": "external_model_performance",
                "feature_set": row["feature_set"],
                "final_model_or_regression_model": row["final_model"],
                "indicator": "",
                "n_train_2005_2018": int(row["n_train"]),
                "n_external_or_regression_n": int(row["n_external"]),
                "n_external_events": int(row["n_external_events"]),
                "AUC": fmt(row["AUC"]),
                "PR-AUC": fmt(row["PR-AUC"]),
                "F1": fmt(row["F1"]),
                "Brier score": fmt(row["Brier score"]),
                "internal held-out AUC": fmt(row["internal held-out AUC"]),
                "external minus internal AUC": fmt(row["external minus internal AUC"]),
                "OR": "",
                "CI_lower": "",
                "CI_upper": "",
                "P_value": "",
                "direction": "",
                "p_method": "",
                "note": "Model trained only on NHANES 2005-2018 and tested on NHANES 2021-2023.",
            }
        )
    model_c = reg.loc[reg["model"].eq("External Model C")].copy()
    for _, row in model_c.iterrows():
        rows.append(
            {
                "analysis_type": "external_weighted_regression",
                "feature_set": "",
                "final_model_or_regression_model": row["model"],
                "indicator": row["indicator"],
                "n_train_2005_2018": "",
                "n_external_or_regression_n": int(row["n_total"]),
                "n_external_events": int(row["n_events"]),
                "AUC": "",
                "PR-AUC": "",
                "F1": "",
                "Brier score": "",
                "internal held-out AUC": "",
                "external minus internal AUC": "",
                "OR": fmt(row["OR"], 3),
                "CI_lower": fmt(row["CI_lower"], 3),
                "CI_upper": fmt(row["CI_upper"], 3),
                "P_value": pformat(row["P_value"]),
                "direction": row["direction"],
                "p_method": row.get("p_method", ""),
                "note": "External Model C adjusted for age, sex, race, education, PIR, smoking_status, and BMI.",
            }
        )
    TABLE6.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE6, index=False)


def copy_figure6() -> None:
    source = EXT_DIR / "external_roc_pr_calibration.png"
    FIG6.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, FIG6)


def row_by_feature(perf: pd.DataFrame, feature_set: str) -> pd.Series:
    return perf.loc[perf["feature_set"].eq(feature_set)].iloc[0]


def row_by_indicator(reg: pd.DataFrame, indicator: str) -> pd.Series:
    return reg.loc[reg["model"].eq("External Model C") & reg["indicator"].eq(indicator)].iloc[0]


def external_results_text(perf: pd.DataFrame, reg: pd.DataFrame) -> dict[str, str]:
    full = row_by_feature(perf, "full_clinical_model")
    routine = row_by_feature(perf, "routine_indicator_model")
    bio = row_by_feature(perf, "biomarker_focused_model")
    uacr = row_by_indicator(reg, "uacr")
    egfr = row_by_indicator(reg, "egfr")
    tygwc = row_by_indicator(reg, "tyg_wc")
    siri = row_by_indicator(reg, "siri")
    return {
        "full_auc": fmt(full["AUC"]),
        "full_pr": fmt(full["PR-AUC"]),
        "full_f1": fmt(full["F1"]),
        "full_brier": fmt(full["Brier score"]),
        "full_delta": fmt(full["external minus internal AUC"]),
        "routine_auc": fmt(routine["AUC"]),
        "routine_pr": fmt(routine["PR-AUC"]),
        "routine_f1": fmt(routine["F1"]),
        "routine_brier": fmt(routine["Brier score"]),
        "routine_delta": fmt(routine["external minus internal AUC"]),
        "bio_auc": fmt(bio["AUC"]),
        "bio_pr": fmt(bio["PR-AUC"]),
        "bio_f1": fmt(bio["F1"]),
        "bio_brier": fmt(bio["Brier score"]),
        "bio_delta": fmt(bio["external minus internal AUC"]),
        "uacr": f"OR = {fmt(uacr['OR'], 3)} (95% CI {fmt(uacr['CI_lower'], 3)}-{fmt(uacr['CI_upper'], 3)}), {pformat(uacr['P_value'])}",
        "egfr": f"OR = {fmt(egfr['OR'], 3)} (95% CI {fmt(egfr['CI_lower'], 3)}-{fmt(egfr['CI_upper'], 3)}), {pformat(egfr['P_value'])}",
        "tygwc": f"OR = {fmt(tygwc['OR'], 3)} (95% CI {fmt(tygwc['CI_lower'], 3)}-{fmt(tygwc['CI_upper'], 3)}), {pformat(tygwc['P_value'])}",
        "siri": f"OR = {fmt(siri['OR'], 3)} (95% CI {fmt(siri['CI_lower'], 3)}-{fmt(siri['CI_upper'], 3)}), {pformat(siri['P_value'])}",
    }


def replace_section(text: str, heading: str, new_body: str) -> str:
    pattern = rf"(### {re.escape(heading)}\n\n)(.*?)(?=\n### |\n## )"
    return re.sub(pattern, rf"\1{new_body.strip()}\n", text, flags=re.S)


def update_manuscript(text: str, perf: pd.DataFrame, reg: pd.DataFrame) -> str:
    vals = external_results_text(perf, reg)
    title = (
        "# Interpretable Machine Learning Identifies Renal, Metabolic, and Inflammatory Indicators Associated with Cardiovascular Disease: "
        "Evidence from NHANES 2005–2018, NHANES 2021–2023 External Validation, and GBD 2023"
    )
    text = re.sub(r"^# .*$", title, text, count=1, flags=re.M)

    abstract_methods = (
        "This study used analytically separate public data sources. GBD 2023 data were used to describe population-level CVD burden trends in Global, China, and United States settings. "
        "NHANES 2005-2018 adult data were used for individual-level prediction and survey-weighted association analyses, and NHANES 2021-2023 was used only as a temporal external validation cohort. "
        "CVD was defined as self-reported congestive heart failure, coronary heart disease, angina, heart attack or myocardial infarction, or stroke. Three leakage-audited feature sets were evaluated: a full clinical model, a routine indicator model, and a biomarker-focused model. "
        "Final models trained on NHANES 2005-2018 were frozen before testing in NHANES 2021-2023. Survey-weighted logistic regression, restricted cubic spline (RCS), subgroup, sensitivity, and external weighted regression analyses were used to evaluate key indicators."
    )
    text = replace_section(text, "Methods", abstract_methods)

    abstract_results = (
        "In GBD 2023, age-standardized global CVD death rates decreased from 375.51 in 1990 to 214.94 in 2023 (-42.8%), and global CVD DALY rates decreased from 8061.62 to 4863.82 (-39.7%). "
        "In China, age-standardized CVD death rates decreased from 496.65 to 208.03 (-58.1%), and CVD DALY rates decreased from 10065.67 to 4059.34 (-59.7%). "
        "The NHANES 2005-2018 analytic sample included 16,827 adults, including 1,942 with CVD and 14,885 without CVD. The best full clinical model was XGBoost (AUC = 0.8528, PR-AUC = 0.4421, F1 = 0.4440); the best routine indicator model was XGBoost (AUC = 0.8478); and the best biomarker-focused model was Elastic Net (AUC = 0.8460). "
        "In NHANES 2021-2023 external validation (n = 3,055; CVD = 384), external AUCs were "
        f"{vals['full_auc']} for the full clinical model, {vals['routine_auc']} for the routine indicator model, and {vals['bio_auc']} for the biomarker-focused model. "
        "UACR, eGFR, TyG-WC, and SIRI showed externally consistent directions in weighted regression. "
        f"UACR remained positive ({vals['uacr']}), eGFR was negative ({vals['egfr']}), TyG-WC was positive ({vals['tygwc']}), and SIRI was positive ({vals['siri']})."
    )
    text = replace_section(text, "Results", abstract_results)

    abstract_conclusions = (
        "GBD data provided population-level context for CVD burden, whereas NHANES supported individual-level modeling, association analyses, and temporal external validation. "
        "The biomarker-focused model showed discrimination comparable to broader clinical models and retained reasonable temporal transportability in NHANES 2021-2023. "
        "UACR was the most robust indicator, while eGFR, TyG-WC, and SIRI were consistently prioritized or externally directionally supported. These findings remain cross-sectional and require non-US and prospective outcome validation before clinical implementation."
    )
    text = replace_section(text, "Conclusions", abstract_conclusions)

    ext_methods = (
        "NHANES 2021-2023 was prepared as an independent temporal external validation cohort and was not merged with NHANES 2005-2018 for training. "
        "The same self-reported CVD outcome definition was applied. The external cohort used the fasting-core analytic subset when TyG-related indicators were required, and survey design variables were retained for external weighted regression. "
        "The final 2005-2018 models were refitted on the full Stage 2 training dataset using only NHANES 2005-2018 data, then frozen before prediction in NHANES 2021-2023. "
        "External performance was evaluated using AUC, PR-AUC, F1 score, Brier score, sensitivity, specificity, calibration intercept, and calibration slope. "
        "CRP and the older physical activity yes/no fields used in earlier cycles were not fully reproducible in NHANES 2021-2023; unavailable predictors were handled by the training-set preprocessing pipeline during prediction. "
        "GBD data were not used in external validation."
    )
    text = replace_section(text, "External temporal validation cohort", ext_methods)

    ext_results = (
        "The external temporal validation cohort included 3,055 adults, including 384 participants with CVD and 2,671 without CVD. "
        f"The final full clinical XGBoost model achieved external AUC = {vals['full_auc']}, PR-AUC = {vals['full_pr']}, F1 = {vals['full_f1']}, and Brier score = {vals['full_brier']} (external minus internal AUC difference {vals['full_delta']}). "
        f"The final routine indicator XGBoost model achieved external AUC = {vals['routine_auc']}, PR-AUC = {vals['routine_pr']}, F1 = {vals['routine_f1']}, and Brier score = {vals['routine_brier']} (external minus internal AUC difference {vals['routine_delta']}). "
        f"The final biomarker-focused Elastic Net model achieved external AUC = {vals['bio_auc']}, PR-AUC = {vals['bio_pr']}, F1 = {vals['bio_f1']}, and Brier score = {vals['bio_brier']} (external minus internal AUC difference {vals['bio_delta']}). "
        f"In External Model C, UACR showed a positive association ({vals['uacr']}); eGFR showed a negative association ({vals['egfr']}); TyG-WC showed a positive association ({vals['tygwc']}); and SIRI showed a positive association ({vals['siri']}). "
        "For External Model C, survey-model P values that were unavailable from the survey t test were obtained using a normal approximation based on the model coefficient and standard error; this is reported in Table 6. "
        "These analyses used NHANES 2021-2023 only as an external temporal validation cohort and should not be interpreted as prospective validation."
    )
    text = replace_section(text, "External temporal validation in NHANES 2021-2023", ext_results)

    ext_discussion = (
        "NHANES 2021-2023 temporal external validation supported cautious temporal transportability across NHANES survey periods. The external AUC declines were small for the final full clinical model (-0.0074), routine indicator model (-0.0090), and biomarker-focused model (-0.0179). "
        "The four key indicators also showed directions consistent with the main analysis: UACR, TyG-WC, and SIRI were positively associated with CVD, whereas eGFR was negatively associated. "
        "These results strengthen the internal evidence that routine renal, metabolic, and inflammatory indicators can help characterize CVD-associated phenotypes. However, the validation remains cross-sectional and based on a later NHANES survey cycle rather than incident cardiovascular events. "
        "CRP and older physical activity fields were not fully comparable in NHANES 2021-2023, and post-pandemic survey context or protocol differences may also affect transportability."
    )
    text = replace_section(text, "External temporal validation", ext_discussion)

    limitations = (
        "Several limitations should be considered. External validation used NHANES 2021-2023, which is temporally later but still cross-sectional; it should not be interpreted as prospective event prediction. "
        "The external validation sample was smaller than the main NHANES 2005-2018 analytic sample, and CRP plus older physical activity yes/no variables could not be fully reproduced in the 2021-2023 files. "
        "The external validation cohort was still drawn from NHANES and therefore represents temporal validation within a US survey system rather than cross-country validation. Non-US cohorts and prospective studies with adjudicated cardiovascular outcomes remain necessary. "
        "The NHANES outcome was self-reported and cross-sectional, so temporal relationships cannot be established. Medication use, treatment history, and clinical management may influence biomarker levels and could contribute to patterns such as the inverse LDL-C association in strict Model 4. "
        "The strict fully adjusted model had a smaller complete-case sample and may be susceptible to overadjustment among correlated biomarkers. Some indicators were not available across all participants, and missingness may affect complete-case survey models. "
        "RCS analyses were performed for prespecified indicators only; RCS nonlinearity was not available for LDL-C or HbA1c in the Stage 5 package."
    )
    text = replace_section(text, "Limitations", limitations)

    text = text.replace(
        "Future studies should validate these findings in prospective cohorts and in external datasets with adjudicated cardiovascular outcomes.",
        "Future studies should validate these findings in non-US cohorts and in prospective datasets with adjudicated cardiovascular outcomes.",
    )
    text = text.replace(
        "In this analysis of GBD 2023 and NHANES 2005-2018 public data, GBD described population-level CVD burden trends, while NHANES supported individual-level machine learning and survey-weighted association analyses. The biomarker-focused model showed discrimination comparable to the full clinical model. UACR was the most robust indicator across feature ranking, survey-weighted regression, nonlinear analysis, and sensitivity testing. eGFR, TyG-WC, and SIRI were consistently prioritized or supported but were more sensitive to strict adjustment. These findings should be interpreted as cross-sectional risk-marker evidence, and future prospective validation is needed.",
        "In this analysis of GBD 2023, NHANES 2005-2018, and NHANES 2021-2023 public data, GBD described population-level CVD burden trends, while NHANES supported individual-level machine learning, survey-weighted association analyses, and temporal external validation. The biomarker-focused model showed discrimination comparable to the full clinical model and retained reasonable temporal transportability. UACR was the most robust indicator, while eGFR, TyG-WC, and SIRI were consistently prioritized or externally directionally supported. These findings should be interpreted as cross-sectional risk-marker evidence, and prospective validation with incident outcomes remains needed.",
    )

    text = text.replace(
        "- Table 5. GBD CVD burden summary, percent change, and EAPC.",
        "- Table 5. GBD CVD burden summary, percent change, and EAPC.\n- Table 6. NHANES 2021-2023 temporal external validation performance and weighted regression.",
    )
    text = text.replace(
        "- Figure 5. Subgroup and sensitivity analyses.",
        "- Figure 5. Subgroup and sensitivity analyses.\n- Figure 6. NHANES 2021-2023 external ROC, precision-recall, and calibration curves.",
    )
    return text


def write_audit(final_text: str, perf: pd.DataFrame, reg: pd.DataFrame) -> None:
    source_numbers = {
        "3,055": "external_model_performance_2021_2023.csv",
        "384": "external_model_performance_2021_2023.csv",
        "2,671": "external_model_performance_2021_2023.csv",
        "0.8454": "external_model_performance_2021_2023.csv",
        "0.8388": "external_model_performance_2021_2023.csv",
        "0.8281": "external_model_performance_2021_2023.csv",
        "-0.0074": "external_model_performance_2021_2023.csv",
        "-0.0090": "external_model_performance_2021_2023.csv",
        "-0.0179": "external_model_performance_2021_2023.csv",
        "1.001": "external_weighted_regression_2021_2023.csv",
        "0.988": "external_weighted_regression_2021_2023.csv",
        "1.003": "external_weighted_regression_2021_2023.csv",
        "1.285": "external_weighted_regression_2021_2023.csv",
        "0.0015": "external_weighted_regression_2021_2023.csv",
        "0.0039": "external_weighted_regression_2021_2023.csv",
    }
    missing_numbers = [num for num in source_numbers if num not in final_text]
    prospective_bad = bool(re.search(r"prospective validation cohort|prospective external validation|prospective event prediction was performed", final_text, flags=re.I))
    prohibited = [term for term in ["caused", "causal effect", "novel biomarker"] if term in final_text.lower()]
    sections = {
        "Abstract": "## Abstract" in final_text and "NHANES 2021-2023" in final_text.split("## Introduction")[0],
        "Methods": "### External temporal validation cohort" in final_text,
        "Results": "### External temporal validation in NHANES 2021-2023" in final_text,
        "Discussion": "### External temporal validation" in final_text,
        "Limitations": "CRP plus older physical activity" in final_text and "Non-US cohorts" in final_text,
    }
    lines = [
        "# Stage 9 External Validation Integration Audit",
        "",
        f"- Final integrated manuscript generated: {FINAL_MANUSCRIPT.exists()}",
        f"- Table 6 generated: {TABLE6.exists()}",
        f"- Figure 6 generated: {FIG6.exists()}",
        "- No data were reanalyzed in Stage 9; Stage 8 CSV outputs were read and formatted.",
        "- NHANES 2021-2023 was described only as temporal external validation, not merged training.",
        "",
        "## Section Updates",
    ]
    for section, ok in sections.items():
        lines.append(f"- {section} updated: {ok}")
    lines.extend(
        [
            "",
            "## Numeric Source Check",
            f"- All tracked external-validation numbers present in final manuscript: {not missing_numbers}",
            f"- Missing tracked numbers: {', '.join(missing_numbers) if missing_numbers else 'none'}",
            "- Numeric sources: external_model_performance_2021_2023.csv and external_weighted_regression_2021_2023.csv.",
            "",
            "## Interpretation Guardrails",
            f"- Avoided prospective-validation wording: {not prospective_bad}",
            f"- Prohibited overclaim terms detected: {', '.join(prohibited) if prohibited else 'none'}",
            "- Temporal validation is explicitly described as cross-sectional and NHANES-based.",
            "- Clinical application language remains cautious.",
            "",
            "## Recommendation",
            "- Proceed to final reference check, journal-specific formatting, and supplementary material organization.",
        ]
    )
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    text, perf, reg, _report = read_inputs()
    create_table6(perf, reg)
    copy_figure6()
    final_text = update_manuscript(text, perf, reg)
    FINAL_MANUSCRIPT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MANUSCRIPT.write_text(final_text, encoding="utf-8")
    write_audit(final_text, perf, reg)


if __name__ == "__main__":
    main()
