from __future__ import annotations

import csv
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_MANUSCRIPT = PROJECT_ROOT / "outputs/manuscript_draft/CVD_NHANES_GBD_ML_manuscript_draft_en_stage9_final_with_external_validation.md"
OUTPUT_MANUSCRIPT = PROJECT_ROOT / "outputs/manuscript_draft/CVD_NHANES_GBD_ML_stage10_with_references.md"
REFERENCE_TABLE = PROJECT_ROOT / "outputs/manuscript_results/reference_verification_table.csv"
TRIPOD_CHECK = PROJECT_ROOT / "outputs/diagnostics/TRIPOD_AI_checklist_mapping.md"
STROBE_CHECK = PROJECT_ROOT / "outputs/diagnostics/STROBE_checklist_mapping.md"
RISK_AUDIT = PROJECT_ROOT / "outputs/diagnostics/stage10_pre_submission_risk_audit.md"
SUPP_DIR = PROJECT_ROOT / "outputs/supplementary"


@dataclass
class Reference:
    citation_number: int
    topic: str
    title: str
    first_author: str
    authors: str
    journal: str
    year: str
    DOI: str
    PMID: str
    URL: str
    verification_status: str
    used_in_section: str

    def row(self) -> dict:
        out = asdict(self)
        out.pop("authors")
        return out

    def formatted(self) -> str:
        authors = self.authors.rstrip(".")
        title = self.title.rstrip(".")
        parts = [f"{self.citation_number}. {authors}.", f"{title}."]
        if self.journal:
            if ";" in self.journal:
                journal, detail = self.journal.split(";", 1)
                parts.append(f"{journal}. {self.year};{detail}.")
            else:
                parts.append(f"{self.journal}. {self.year}.")
        elif self.year and self.year != "not available":
            parts.append(f"{self.year}.")
        if self.DOI:
            parts.append(f"doi:{self.DOI}.")
        if self.PMID:
            parts.append(f"PMID:{self.PMID}.")
        if self.URL:
            parts.append(f"Available from: {self.URL}.")
        return " ".join(parts)


REFERENCES = [
    Reference(1, "GBD 2023 methods and cardiovascular disease burden", "Global, Regional, and National Burden of Cardiovascular Diseases and Risk Factors in 204 Countries and Territories, 1990-2023", "Global Burden of Cardiovascular Diseases and Risks 2023 Collaborators", "Global Burden of Cardiovascular Diseases and Risks 2023 Collaborators", "Journal of the American College of Cardiology;86(22):2167-2243", "2025", "10.1016/j.jacc.2025.08.015", "40990886", "https://www.sciencedirect.com/science/article/pii/S0735109725074285", "VERIFIED_PUBLISHER_AND_PUBMED_SEARCH", "Introduction; Methods; Results; Discussion"),
    Reference(2, "GBD Results Tool / IHME data source", "GBD Results", "Institute for Health Metrics and Evaluation", "Institute for Health Metrics and Evaluation", "", "not available", "", "", "https://www.healthdata.org/data-tools-practices/interactive-visuals/gbd-results", "VERIFIED_OFFICIAL_SOURCE", "Methods; Data availability"),
    Reference(3, "NHANES survey design and analytic guidelines", "NHANES Survey Methods and Analytic Guidelines", "Centers for Disease Control and Prevention", "Centers for Disease Control and Prevention, National Center for Health Statistics", "", "not available", "", "", "https://wwwn.cdc.gov/nchs/nhanes/analyticguidelines.aspx", "VERIFIED_OFFICIAL_SOURCE", "Methods; Ethics"),
    Reference(4, "NHANES 2021-2023 analytic guidance", "Brief Overview of Sample Design, Nonresponse Bias Assessment, and Analytic Guidelines for NHANES August 2021-August 2023", "Centers for Disease Control and Prevention", "Centers for Disease Control and Prevention, National Center for Health Statistics", "", "2024", "", "", "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/overviewbrief.aspx?Cycle=2021-2023", "VERIFIED_OFFICIAL_SOURCE", "Methods; External validation; Limitations"),
    Reference(5, "NHANES 2021-2023 data cycle documentation", "NHANES Questionnaires, Datasets, and Related Documentation: August 2021-August 2023", "Centers for Disease Control and Prevention", "Centers for Disease Control and Prevention, National Center for Health Statistics", "", "not available", "", "", "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023", "VERIFIED_OFFICIAL_SOURCE", "Methods; External validation"),
    Reference(6, "NHANES cardiovascular questionnaire and self-reported disease variables", "NHANES 2021-2023 Data Documentation, Codebook, and Frequencies: Medical Conditions (MCQ_L)", "Centers for Disease Control and Prevention", "Centers for Disease Control and Prevention, National Center for Health Statistics", "", "2021-2023", "", "", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.htm", "VERIFIED_OFFICIAL_SOURCE", "Methods; Outcome definition"),
    Reference(7, "CKD-EPI eGFR equation", "A new equation to estimate glomerular filtration rate", "Levey", "Levey AS, Stevens LA, Schmid CH, Zhang YL, Castro AF 3rd, Feldman HI, et al.; CKD-EPI", "Annals of Internal Medicine;150(9):604-612", "2009", "10.7326/0003-4819-150-9-200905050-00006", "19414839", "https://pubmed.ncbi.nlm.nih.gov/19414839/", "VERIFIED_PUBMED", "Introduction; Methods; Discussion"),
    Reference(8, "UACR/eGFR and cardiovascular disease risk", "Association of estimated glomerular filtration rate and albuminuria with all-cause and cardiovascular mortality in general population cohorts: a collaborative meta-analysis", "Matsushita", "Matsushita K, van der Velde M, Astor BC, Woodward M, Levey AS, de Jong PE, et al.; Chronic Kidney Disease Prognosis Consortium", "The Lancet;375(9731):2073-2081", "2010", "10.1016/S0140-6736(10)60674-5", "20483451", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3993088/", "VERIFIED_PMC_PUBMED", "Introduction; Discussion"),
    Reference(9, "TyG index definition and insulin resistance validation", "The product of triglycerides and glucose, a simple measure of insulin sensitivity. Comparison with the euglycemic-hyperinsulinemic clamp", "Guerrero-Romero", "Guerrero-Romero F, Simental-Mendia LE, Gonzalez-Ortiz M, Martinez-Abundis E, Ramos-Zavala MG, Hernandez-Gonzalez SO, et al.", "Journal of Clinical Endocrinology and Metabolism;95(7):3347-3351", "2010", "10.1210/jc.2010-0288", "20484475", "https://pubmed.ncbi.nlm.nih.gov/20484475/", "VERIFIED_PUBMED_SEARCH", "Introduction; Methods"),
    Reference(10, "TyG index / TyG-WC and cardiovascular disease", "The association between triglyceride-glucose index and its combination with obesity indicators and cardiovascular disease: NHANES 2003-2018", "Dang", "Dang K, Wang X, Hu J, Zhang Y, Cheng L, Qi X, et al.", "Cardiovascular Diabetology;23(1):8", "2024", "10.1186/s12933-023-02115-9", "38184598", "https://link.springer.com/article/10.1186/s12933-023-02115-9", "VERIFIED_PUBLISHER_AND_PUBMED", "Introduction; Discussion"),
    Reference(11, "SIRI / inflammatory indices and cardiovascular disease", "The relationship between system inflammation response index and coronary heart disease: a cross-sectional study (NHANES 2007-2016)", "Zhang", "Zhang TY, Chen HL, Shi Y, Jin Y, Zhang Y, Chen Y", "Frontiers in Cardiovascular Medicine;11:1439913", "2024", "10.3389/fcvm.2024.1439913", "", "https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1439913/full", "VERIFIED_PUBLISHER_NO_PMID_FOUND", "Introduction; Discussion"),
    Reference(12, "XGBoost methodology", "XGBoost: A Scalable Tree Boosting System", "Chen", "Chen T, Guestrin C", "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining;785-794", "2016", "10.1145/2939672.2939785", "", "https://arxiv.org/abs/1603.02754", "VERIFIED_DOI_AND_ARXIV", "Methods"),
    Reference(13, "Elastic Net methodology", "Regularization and variable selection via the elastic net", "Zou", "Zou H, Hastie T", "Journal of the Royal Statistical Society Series B: Statistical Methodology;67(2):301-320", "2005", "10.1111/j.1467-9868.2005.00503.x", "", "https://academic.oup.com/jrsssb/article/67/2/301/7109482", "VERIFIED_PUBLISHER", "Methods"),
    Reference(14, "SHAP methodology", "A Unified Approach to Interpreting Model Predictions", "Lundberg", "Lundberg SM, Lee SI", "Advances in Neural Information Processing Systems 30", "2017", "", "", "https://papers.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions", "VERIFIED_CONFERENCE_SOURCE", "Introduction; Methods"),
    Reference(15, "Tree SHAP methodology", "From local explanations to global understanding with explainable AI for trees", "Lundberg", "Lundberg SM, Erion G, Chen H, DeGrave A, Prutkin JM, Nair B, et al.", "Nature Machine Intelligence;2(1):56-67", "2020", "10.1038/s42256-019-0138-9", "", "https://www.nature.com/articles/s42256-019-0138-9", "VERIFIED_PUBLISHER", "Introduction; Methods"),
    Reference(16, "Survey-weighted logistic regression / complex sampling analysis", "Analysis of Complex Survey Samples", "Lumley", "Lumley T", "Journal of Statistical Software;9(8):1-19", "2004", "10.18637/jss.v009.i08", "", "https://www.jstatsoft.org/article/view/v009i08", "VERIFIED_JOURNAL_SOURCE", "Methods"),
    Reference(17, "Restricted cubic spline methodology", "Flexible regression models with cubic splines", "Durrleman", "Durrleman S, Simon R", "Statistics in Medicine;8(5):551-561", "1989", "10.1002/sim.4780080504", "2657958", "https://pubmed.ncbi.nlm.nih.gov/2657958/", "VERIFIED_PUBMED", "Methods"),
    Reference(18, "TRIPOD+AI reporting guideline", "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods", "Collins", "Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al.", "BMJ;385:e078378", "2024", "10.1136/bmj-2023-078378", "38626948", "https://pubmed.ncbi.nlm.nih.gov/38626948/", "VERIFIED_PUBMED", "Methods; Reporting checklist"),
    Reference(19, "STROBE reporting guideline", "The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies", "von Elm", "von Elm E, Altman DG, Egger M, Pocock SJ, Gotzsche PC, Vandenbroucke JP; STROBE Initiative", "The Lancet;370(9596):1453-1457", "2007", "10.1016/S0140-6736(07)61602-X", "18064739", "https://pubmed.ncbi.nlm.nih.gov/18064739/", "VERIFIED_PUBMED", "Methods; Reporting checklist"),
]


def pformat_numeric(raw: str) -> str:
    value = float(raw)
    return "P < 0.001" if value < 0.001 else f"P = {value:.4f}"


def clean_numeric_formatting(text: str) -> str:
    num = r"([0-9]+(?:\.[0-9]+)?)"
    text = re.sub(r"\bAUC\s*=\s*" + num, lambda m: f"AUC = {float(m.group(1)):.4f}", text)
    text = re.sub(r"\bPR-AUC\s*=\s*" + num, lambda m: f"PR-AUC = {float(m.group(1)):.4f}", text)
    text = re.sub(r"\bF1\s*=\s*" + num, lambda m: f"F1 = {float(m.group(1)):.4f}", text)
    text = re.sub(r"\bBrier score\s*=\s*" + num, lambda m: f"Brier score = {float(m.group(1)):.4f}", text)
    text = re.sub(r"\bOR\s*=\s*" + num, lambda m: f"OR = {float(m.group(1)):.3f}", text)
    text = re.sub(r"95% CI ([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)", lambda m: f"95% CI {float(m.group(1)):.3f}-{float(m.group(2)):.3f}", text)
    text = re.sub(r"\bP\s*=\s*([0-9.]+e[-+]?\d+|[0-9]+(?:\.[0-9]+)?)", lambda m: pformat_numeric(m.group(1)), text, flags=re.I)
    return text


def add_stage10_citations(text: str) -> str:
    # Replace old Stage 7 numbering blocks first.
    replacements = {
        "[1,2]": "[1,2]",
        "[3,4]": "[3,4]",
        "[5]": "[6]",
        "[6,7]": "[7,8]",
        "[6-10]": "[7-11]",
        "[6,8-10]": "[7,9-11]",
        "[7]": "[8]",
        "[8,9]": "[9,10]",
        "[9]": "[10]",
        "[10]": "[11]",
        "[11-13]": "[12,14,15]",
        "[11]": "[12]",
        "[12,13]": "[14,15]",
        "[14]": "[16]",
        "[15]": "[17]",
    }
    placeholders = {}
    for idx, (old, new) in enumerate(sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)):
        token = f"@@CIT{idx}@@"
        placeholders[token] = new
        text = text.replace(old, token)
    for token, new in placeholders.items():
        text = text.replace(token, new)
    text = text.replace(
        "NHANES 2021-2023 was prepared as an independent temporal external validation cohort",
        "NHANES 2021-2023 was prepared as an independent temporal external validation cohort using CDC/NCHS documentation and analytic guidance [4,5]",
    )
    text = text.replace(
        "Machine learning analyses compared logistic regression, random forest, XGBoost, LightGBM, support vector machine, and elastic net classifiers across the three feature sets [12].",
        "Machine learning analyses compared logistic regression, random forest, XGBoost, LightGBM, support vector machine, and elastic net classifiers across the three feature sets [12,13].",
    )
    text = text.replace(
        "Derived indicators included TyG index, TyG-BMI, TyG-WC, atherogenic index of plasma, non-HDL-C, neutrophil-to-lymphocyte ratio, platelet-to-lymphocyte ratio, systemic immune-inflammation index, SIRI, eGFR, and UACR when source variables were available [7,9-11].",
        "Derived indicators included TyG index, TyG-BMI, TyG-WC, atherogenic index of plasma, non-HDL-C, neutrophil-to-lymphocyte ratio, platelet-to-lymphocyte ratio, systemic immune-inflammation index, SIRI, eGFR, and UACR when source variables were available [7-11].",
    )
    text = text.replace(
        "Data preparation, machine learning, feature ranking, GBD analysis, and manuscript table generation were performed using Python.",
        "The reporting structure was checked against TRIPOD+AI for prediction modeling and STROBE for observational epidemiology [18,19]. Data preparation, machine learning, feature ranking, GBD analysis, and manuscript table generation were performed using Python.",
    )
    return text


def replace_references(text: str) -> str:
    refs = "\n".join(ref.formatted() for ref in REFERENCES)
    return re.sub(r"## References\n[\s\S]*$", "## References\n\n" + refs + "\n", text)


def write_reference_table() -> None:
    REFERENCE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with REFERENCE_TABLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(REFERENCES[0].row().keys()))
        writer.writeheader()
        for ref in REFERENCES:
            writer.writerow(ref.row())


def write_checklist(path: Path, title: str, rows: list[tuple[str, str, str]]) -> None:
    lines = [f"# {title}", "", "| Item | Status | Mapping / comment |", "|---|---|---|"]
    lines.extend(f"| {item} | {status} | {comment} |" for item, status, comment in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tripod_checklist() -> None:
    rows = [
        ("Data source", "COMPLETE", "GBD, NHANES 2005-2018, and NHANES 2021-2023 are described separately."),
        ("Study design", "COMPLETE", "Public database study with population-level GBD context and individual-level NHANES modeling."),
        ("Target outcome definition", "COMPLETE", "Self-reported CHF, CHD, angina, heart attack, or stroke; outcome fields excluded from predictors."),
        ("Predictor definition", "COMPLETE", "Predictor domains and derived indicators are described; feature sets are named."),
        ("Missing value handling", "COMPLETE", "Training-set pipeline imputation and external unavailable CRP/PAQ notes are reported."),
        ("Training/test split", "COMPLETE", "Stratified 70:30 internal split reported."),
        ("External validation cohort", "COMPLETE", "NHANES 2021-2023 temporal external validation reported as cross-sectional, not prospective."),
        ("Model types", "COMPLETE", "Logistic regression, random forest, XGBoost, LightGBM, SVM, and Elastic Net listed."),
        ("Model tuning / hyperparameters", "PARTIAL", "Final model classes and scripts are available; full hyperparameter grid is not manuscript-level detail."),
        ("Performance metrics", "COMPLETE", "AUC, PR-AUC, F1, Brier score, sensitivity, specificity, and calibration metrics are reported."),
        ("Calibration", "COMPLETE", "Calibration curves and external calibration intercept/slope are generated."),
        ("Overfitting prevention", "COMPLETE", "Leakage audit, held-out testing, and external temporal validation described."),
        ("Interpretability", "COMPLETE", "SHAP, permutation, model importance, and consensus ranking are described."),
        ("Reproducibility", "PARTIAL", "Scripts and environment files exist; final model binary sharing policy requires journal/repository decision."),
        ("Limitations", "COMPLETE", "Cross-sectional design, self-report outcome, correlated biomarkers, and external validation limits are stated."),
    ]
    write_checklist(TRIPOD_CHECK, "TRIPOD+AI Checklist Mapping", rows)


def write_strobe_checklist() -> None:
    rows = [
        ("Title and abstract identify design", "COMPLETE", "Public-database and cross-sectional NHANES association framing is present."),
        ("Background and rationale", "COMPLETE", "CVD burden and renal/metabolic/inflammatory rationale described."),
        ("Objectives", "COMPLETE", "Prediction, feature prioritization, survey-weighted association, and external temporal validation objectives stated."),
        ("Data sources", "COMPLETE", "GBD and NHANES data sources described; analytic levels kept separate."),
        ("Participants / study subjects", "COMPLETE", "Adult NHANES sample and external 2021-2023 cohort described."),
        ("Variables", "COMPLETE", "CVD outcome, candidate predictors, derived indicators, and covariate models specified."),
        ("Bias control", "PARTIAL", "Leakage audit and survey design described; residual self-report and medication confounding remain."),
        ("Sample size", "COMPLETE", "Main and external sample sizes and event counts reported."),
        ("Statistical methods", "COMPLETE", "Survey-weighted regression, RCS, subgroup, sensitivity, and ML metrics described."),
        ("Descriptive results", "COMPLETE", "Weighted baseline characteristics table generated."),
        ("Main results", "COMPLETE", "Model performance, weighted regression, RCS, subgroup, sensitivity, and external validation results included."),
        ("Sensitivity analyses", "COMPLETE", "Sensitivity analyses table generated and summarized."),
        ("Discussion", "COMPLETE", "Main findings and cautious interpretation included."),
        ("Limitations", "COMPLETE", "Cross-sectional design, external validation limits, missing comparability, and overadjustment covered."),
        ("Generalisability", "COMPLETE", "US NHANES and non-US validation limits stated."),
        ("Funding", "NEED_PROJECT_CONFIRMATION", "Funding and author-specific declarations remain placeholders."),
    ]
    write_checklist(STROBE_CHECK, "STROBE Checklist Mapping", rows)


def copy_or_placeholder(source: Path, destination: Path, placeholder_columns: Iterable[str]) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists() and source.stat().st_size > 0:
        shutil.copy2(source, destination)
        return True
    pd.DataFrame([{"status": "not available", "expected_source": str(source)}], columns=list(placeholder_columns)).to_csv(destination, index=False)
    return False


def write_supplementary() -> dict[str, bool]:
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    status = {}
    status["Supplementary_Methods.md"] = True
    (SUPP_DIR / "Supplementary_Methods.md").write_text(
        "# Supplementary Methods\n\n"
        "This supplement documents the reproducible public-database workflow. NHANES 2005-2018 was used for model development and survey-weighted association analyses; NHANES 2021-2023 was used only for temporal external validation. GBD 2023 was used only for population-level burden context. CVD was defined from self-reported congestive heart failure, coronary heart disease, angina, heart attack/myocardial infarction, or stroke. Outcome component variables were excluded from predictors. Machine learning preprocessing, including imputation and scaling, was fitted within training-set pipelines. External validation models were trained only on NHANES 2005-2018 and tested in NHANES 2021-2023. Survey-weighted analyses used NHANES design variables and weights as described in the main manuscript.\n",
        encoding="utf-8",
    )
    copies = {
        "Supplementary_Table_S1_variable_mapping.csv": PROJECT_ROOT / "outputs/diagnostics/nhanes_variable_map.csv",
        "Supplementary_Table_S2_missingness.csv": PROJECT_ROOT / "outputs/diagnostics/missingness_2005_2018.csv",
        "Supplementary_Table_S3_full_model_performance.csv": PROJECT_ROOT / "outputs/tables/model_performance_2005_2018_by_featureset.csv",
        "Supplementary_Table_S4_full_feature_ranking.csv": PROJECT_ROOT / "outputs/tables/feature_ranking_2005_2018_by_featureset.csv",
        "Supplementary_Table_S5_sensitivity_analysis.csv": PROJECT_ROOT / "outputs/tables/sensitivity_analysis.csv",
        "Supplementary_Table_S6_external_variable_mapping.csv": PROJECT_ROOT / "outputs/external_validation/nhanes_2021_2023_variable_mapping.csv",
    }
    for name, source in copies.items():
        status[name] = copy_or_placeholder(source, SUPP_DIR / name, ["status", "expected_source"])
    fig_source = PROJECT_ROOT / "outputs/external_validation/external_roc_pr_calibration.png"
    fig_dest = SUPP_DIR / "Supplementary_Figure_S1_external_roc_pr_calibration.png"
    if fig_source.exists():
        shutil.copy2(fig_source, fig_dest)
        status["Supplementary_Figure_S1_external_roc_pr_calibration.png"] = True
    else:
        status["Supplementary_Figure_S1_external_roc_pr_calibration.png"] = False
    return status


def write_risk_audit(text: str, supp_status: dict[str, bool]) -> None:
    need_manual = [r for r in REFERENCES if "NEED_MANUAL_VERIFICATION" in r.verification_status]
    lower = text.lower()
    causal_terms = [term for term in ["caused", "causal effect", "novel biomarker"] if term in lower]
    prospective_mislabel = bool(re.search(r"prospective validation cohort|prospective external validation|prospective event prediction was performed", text, flags=re.I))
    gbd_confusion = bool(re.search(r"GBD (validated|validates|confirmed|confirms) NHANES|GBD.*biomarker validation", text, flags=re.I))
    model4_cautious = "strict fully adjusted model" in text and "overadjustment" in text
    ldl_c_cautious = "LDL-C showed an inverse association" in text and "interpreted cautiously" in text
    tables = {f"Table {i}": f"Table {i}" in text for i in range(1, 7)}
    figures = {f"Figure {i}": f"Figure {i}" in text for i in range(1, 7)}
    lines = [
        "# Stage 10 Pre-Submission Risk Audit",
        "",
        f"1. Unverified references present: {len(need_manual) > 0}",
        f"   - Count: {len(need_manual)}",
        f"2. Overstated causal language detected: {bool(causal_terms)}",
        f"   - Terms: {', '.join(causal_terms) if causal_terms else 'none'}",
        f"3. GBD/NHANES level confusion detected: {gbd_confusion}",
        f"4. Temporal external validation mislabeled as prospective validation: {prospective_mislabel}",
        f"5. Model 4 over-interpretation risk controlled: {model4_cautious}",
        f"6. LDL-C inverse OR cautiously handled: {ldl_c_cautious}",
        f"7. All main figures referenced/listed: {all(figures.values())}",
        f"   - {figures}",
        f"8. All main tables referenced/listed: {all(tables.values())}",
        f"   - {tables}",
        "9. Author information, funding, competing interests, contributions, and final ethics wording still require author confirmation.",
        f"10. Supplementary materials complete: {all(supp_status.values())}",
        f"    - {supp_status}",
        "",
        "## Recommendation",
        "- Proceed to Stage 11 target journal formatting after author/funding/declaration details and any journal-specific word/reference limits are confirmed.",
    ]
    RISK_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    RISK_AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    text = INPUT_MANUSCRIPT.read_text(encoding="utf-8")
    text = add_stage10_citations(text)
    text = clean_numeric_formatting(text)
    text = replace_references(text)
    OUTPUT_MANUSCRIPT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANUSCRIPT.write_text(text, encoding="utf-8")
    write_reference_table()
    write_tripod_checklist()
    write_strobe_checklist()
    supp_status = write_supplementary()
    write_risk_audit(text, supp_status)
    print(f"references_added={len(REFERENCES)}")
    print(f"manual_verification_needed={sum('NEED_MANUAL_VERIFICATION' in r.verification_status for r in REFERENCES)}")
    print(f"tripod_status=PARTIAL (see {TRIPOD_CHECK})")
    print(f"strobe_status=PARTIAL (see {STROBE_CHECK})")
    print(f"supplementary_complete={all(supp_status.values())}")
    print("ready_for_target_journal_formatting=yes_after_author_funding_declaration_confirmation")


if __name__ == "__main__":
    main()
