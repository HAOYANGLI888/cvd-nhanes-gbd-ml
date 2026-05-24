from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = PROJECT_ROOT / "outputs/external_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Stage 8 external validation report and manuscript update.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def fmt(x, digits=4) -> str:
    if pd.isna(x):
        return "not available"
    return f"{float(x):.{digits}f}"


def pformat(x) -> str:
    if pd.isna(x):
        return "not available"
    x = float(x)
    return "P < 0.001" if x < 0.001 else f"P = {x:.4f}"


def load_context(root: Path) -> Dict[str, pd.DataFrame]:
    ext = root / "outputs/external_validation"
    return {
        "flow": read_csv(ext / "external_sample_size_flow.csv"),
        "dist": read_csv(ext / "external_cvd_distribution.csv"),
        "missing": read_csv(ext / "external_missingness_report.csv"),
        "availability": read_csv(ext / "external_key_variable_availability.csv"),
        "mapping": read_csv(ext / "nhanes_2021_2023_variable_mapping.csv"),
        "module_status": read_csv(ext / "external_module_download_status.csv"),
        "perf": read_csv(ext / "external_model_performance_2021_2023.csv"),
        "reg": read_csv(ext / "external_weighted_regression_2021_2023.csv"),
        "missing_features": read_csv(ext / "external_model_missing_features.csv"),
    }


def sample_numbers(flow: pd.DataFrame, dist: pd.DataFrame) -> Dict[str, int]:
    final_n = 0
    if not flow.empty:
        final_n = int(flow.iloc[-1]["remaining_n"])
    cvd = 0
    noncvd = 0
    if not dist.empty:
        for row in dist.itertuples(index=False):
            label = str(getattr(row, "label", ""))
            n = int(getattr(row, "n", 0))
            if label == "CVD":
                cvd = n
            elif label == "Non-CVD":
                noncvd = n
    return {"n": final_n, "cvd": cvd, "noncvd": noncvd}


def direction_summary(reg: pd.DataFrame) -> Dict[str, str]:
    out = {}
    if reg.empty:
        return out
    focus = reg.loc[reg["model"].eq("External Model C")].copy()
    for row in focus.itertuples(index=False):
        indicator = str(row.indicator)
        direction = str(row.direction)
        out[indicator] = f"{direction}; OR = {fmt(row.OR, 3)} (95% CI {fmt(row.CI_lower, 3)}-{fmt(row.CI_upper, 3)}), {pformat(row.P_value)}"
    return out


def direction_sentence(indicator_label: str, text: str) -> str:
    if text == "not available":
        return f"{indicator_label} was not available"
    direction, detail = text.split("; ", 1)
    return f"{indicator_label} showed a {direction} association ({detail})"


def write_report(root: Path, ctx: Dict[str, pd.DataFrame]) -> Path:
    ext = root / "outputs/external_validation"
    numbers = sample_numbers(ctx["flow"], ctx["dist"])
    perf = ctx["perf"]
    reg = ctx["reg"]
    availability = ctx["availability"]
    module_status = ctx["module_status"]

    loaded_modules = []
    failed_modules = []
    if not module_status.empty:
        loaded_modules = module_status.loc[module_status["status"].eq("loaded"), "module"].astype(str).tolist()
        failed_modules = module_status.loc[~module_status["status"].eq("loaded"), "module"].astype(str).tolist()

    available_vars = []
    missing_vars = []
    if not availability.empty:
        available_vars = availability.loc[availability["missing_rate"].lt(1.0), "variable"].astype(str).tolist()
        missing_vars = availability.loc[availability["missing_rate"].ge(1.0), "variable"].astype(str).tolist()

    lines = [
        "# Stage 8 External Temporal Validation Report",
        "",
        "## Data Acquisition",
        f"- NHANES 2021-2023 data successfully downloaded/loaded: {bool(loaded_modules)}",
        f"- Loaded modules: {', '.join(loaded_modules) if loaded_modules else 'not available'}",
        f"- Missing/unloaded modules: {', '.join(failed_modules) if failed_modules else 'none'}",
        "- GBD data were not used in external validation.",
        "- NHANES 2021-2023 was not merged with 2005-2018 for model training.",
        "",
        "## External Cohort",
        f"- Final external validation sample size: {numbers['n']}",
        f"- CVD events: {numbers['cvd']}",
        f"- Non-CVD participants: {numbers['noncvd']}",
        "",
        "## Variable Availability",
        f"- Main-analysis variables available or partly available in 2021-2023: {', '.join(available_vars) if available_vars else 'not available'}",
        f"- Variables unavailable or fully missing in 2021-2023: {', '.join(missing_vars) if missing_vars else 'none among tracked key variables'}",
        "- Notable comparability notes: CRP was unavailable in the loaded 2021-2023 modules, and older PAQ vigorous/moderate work/recreation yes-no fields were not directly reproduced from PAQ_L; models handled unavailable predictors through training-set pipeline imputation.",
        "",
        "## External Model Performance",
    ]
    if perf.empty:
        lines.append("- External model performance file was not generated.")
    else:
        for _, row in perf.iterrows():
            lines.append(
                f"- `{row['feature_set']}` ({row['final_model']}): AUC={fmt(row['AUC'])}, PR-AUC={fmt(row['PR-AUC'])}, "
                f"F1={fmt(row['F1'])}, Brier={fmt(row['Brier score'])}, "
                f"external-internal AUC difference={fmt(row['external minus internal AUC'])}."
            )
    lines.extend(["", "## External Survey-Weighted Regression Direction"])
    directions = direction_summary(reg)
    for indicator in ["uacr", "egfr", "tyg_wc", "siri"]:
        lines.append(f"- `{indicator}`: {directions.get(indicator, 'not available')}")
    if not reg.empty and "p_method" in reg.columns:
        p_methods = ", ".join(sorted(set(reg["p_method"].dropna().astype(str))))
        lines.append(f"- P-value methods recorded in regression output: {p_methods}")

    support_generalization = False
    if not perf.empty and perf["AUC"].notna().any():
        support_generalization = bool((perf["AUC"] >= 0.75).all() and (perf["external minus internal AUC"].abs() <= 0.15).all())
    lines.extend(
        [
            "",
            "## Interpretation",
            f"- Supports model generalization: {'yes, with caution' if support_generalization else 'limited or requires cautious interpretation'}",
            "- Recommendation for manuscript Results: include if sample size, event count, and external AUCs are acceptable; otherwise present as supplementary temporal validation and discuss limitations.",
            "- If any external performance is materially lower than internal held-out performance, do not force favorable wording; describe the direction and magnitude directly.",
        ]
    )
    report_path = ext / "stage8_external_validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def update_manuscript(root: Path, ctx: Dict[str, pd.DataFrame]) -> Path:
    source = root / "outputs/manuscript_draft/CVD_NHANES_GBD_ML_manuscript_draft_en_stage7_refs.md"
    if not source.exists():
        source = root / "outputs/manuscript_draft/CVD_NHANES_GBD_ML_manuscript_draft_en.md"
    text = source.read_text(encoding="utf-8")
    numbers = sample_numbers(ctx["flow"], ctx["dist"])
    perf = ctx["perf"]
    reg = ctx["reg"]
    directions = direction_summary(reg)

    perf_sentences = []
    display_names = {
        "full_clinical_model": "full clinical",
        "routine_indicator_model": "routine indicator",
        "biomarker_focused_model": "biomarker-focused",
    }
    if not perf.empty:
        for _, row in perf.iterrows():
            display = display_names.get(row["feature_set"], row["feature_set"])
            perf_sentences.append(
                f"The final {display} {row['final_model']} model achieved external AUC = {fmt(row['AUC'])}, "
                f"PR-AUC = {fmt(row['PR-AUC'])}, F1 = {fmt(row['F1'])}, and Brier score = {fmt(row['Brier score'])} "
                f"(external minus internal AUC difference {fmt(row['external minus internal AUC'])})."
            )
    regression_sentence = (
        "In External Model C, "
        + "; ".join(
            [
                direction_sentence("UACR", directions.get("uacr", "not available")),
                direction_sentence("eGFR", directions.get("egfr", "not available")),
                direction_sentence("TyG-WC", directions.get("tyg_wc", "not available")),
                direction_sentence("SIRI", directions.get("siri", "not available")),
            ]
        )
        + "."
    )

    methods_insert = f"""
### External temporal validation cohort

NHANES 2021-2023 was prepared as an independent temporal external validation cohort and was not merged with NHANES 2005-2018 for training. The same self-reported CVD outcome definition was applied. The external cohort used the fasting-core analytic subset when TyG-related indicators were required, and survey design variables were retained for external weighted regression. The final 2005-2018 models were refitted on the full Stage 2 training dataset using only 2005-2018 data and then frozen before prediction in NHANES 2021-2023. External performance was evaluated using AUC, PR-AUC, F1 score, Brier score, sensitivity, specificity, calibration intercept, and calibration slope.
"""
    if "### External temporal validation cohort" not in text:
        text = text.replace("### CVD outcome definition", methods_insert + "\n### CVD outcome definition")

    results_insert = f"""
### External temporal validation in NHANES 2021-2023

The external temporal validation cohort included {numbers['n']:,} adults, including {numbers['cvd']:,} participants with CVD and {numbers['noncvd']:,} without CVD. {' '.join(perf_sentences)} {regression_sentence} These analyses used NHANES 2021-2023 only as an external validation cohort and should not be interpreted as prospective validation.
"""
    if "### External temporal validation in NHANES 2021-2023" not in text:
        text = text.replace("### SHAP and feature importance", results_insert + "\n### SHAP and feature importance")

    discussion_insert = """
### External temporal validation

The NHANES 2021-2023 temporal validation provides an additional check of model transportability across survey periods. Because the external dataset was prepared after the 2005-2018 model development and was not used for training, it helps assess whether the final models retain discrimination in a later NHANES cohort. The validation remains cross-sectional and should not be described as prospective validation. Differences in module availability, variable definitions, examination protocols, and post-pandemic sampling context may contribute to changes in performance or regression estimates.
"""
    if "### External temporal validation\n" not in text:
        text = text.replace("### Clinical and public health implications", discussion_insert + "\n### Clinical and public health implications")

    limitation_extra = " External validation used NHANES 2021-2023, which is temporally later but still cross-sectional; variable availability and protocol changes may limit direct comparability with 2005-2018."
    if limitation_extra.strip() not in text:
        text = text.replace("Several limitations should be considered.", "Several limitations should be considered." + limitation_extra)
    text = text.replace(
        "Finally, machine learning performance was evaluated in held-out NHANES data and requires external and prospective validation.",
        "Finally, machine learning performance was evaluated in held-out NHANES data and a later NHANES temporal validation cohort, but prospective validation remains necessary.",
    )

    output = root / "outputs/manuscript_draft/CVD_NHANES_GBD_ML_manuscript_draft_en_stage8_external_validation.md"
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ctx = load_context(root)
    write_report(root, ctx)
    update_manuscript(root, ctx)


if __name__ == "__main__":
    main()
