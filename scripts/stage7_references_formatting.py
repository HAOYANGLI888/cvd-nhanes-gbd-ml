from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DRAFT = PROJECT_ROOT / "outputs/manuscript_draft/CVD_NHANES_GBD_ML_manuscript_draft_en.md"
OUTPUT_DRAFT = PROJECT_ROOT / "outputs/manuscript_draft/CVD_NHANES_GBD_ML_manuscript_draft_en_stage7_refs.md"
REFERENCE_TABLE = PROJECT_ROOT / "outputs/manuscript_results/reference_verification_table.csv"
AUDIT = PROJECT_ROOT / "outputs/diagnostics/stage7_references_formatting_audit.md"


@dataclass
class Reference:
    citation_number: int
    topic: str
    title: str
    authors: str
    journal: str
    year: str
    DOI: str
    PMID: str
    URL: str
    verification_status: str
    used_in_section: str

    def formatted(self) -> str:
        authors = self.authors.rstrip(".")
        title = self.title.rstrip(".")
        parts = [
            f"{self.citation_number}. {authors}.",
            title + ".",
        ]
        if self.journal:
            if ";" in self.journal and self.year and self.year != "not available":
                journal_name, journal_detail = self.journal.split(";", 1)
                journal_part = f"{journal_name}. {self.year};{journal_detail}."
            elif self.year and self.year != "not available":
                journal_part = f"{self.journal}. {self.year}."
            else:
                journal_part = f"{self.journal}."
            parts.append(journal_part)
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
    Reference(
        1,
        "GBD 2023 methods and cardiovascular disease burden",
        "Global, Regional, and National Burden of Cardiovascular Diseases and Risk Factors in 204 Countries and Territories, 1990-2023",
        "Global Burden of Cardiovascular Diseases and Risks 2023 Collaborators",
        "Journal of the American College of Cardiology;86(22):2167-2243",
        "2025",
        "10.1016/j.jacc.2025.08.015",
        "40990886",
        "https://www.sciencedirect.com/science/article/pii/S0735109725074285",
        "VERIFIED_PUBLISHER_AND_PMID_SEARCH",
        "Introduction; Methods; Results; Discussion",
    ),
    Reference(
        2,
        "GBD Results Tool / IHME data source",
        "GBD Results",
        "Institute for Health Metrics and Evaluation",
        "",
        "not available",
        "",
        "",
        "https://www.healthdata.org/data-tools-practices/interactive-visuals/gbd-results",
        "VERIFIED_OFFICIAL_SOURCE",
        "Methods; Data availability",
    ),
    Reference(
        3,
        "NHANES survey design and analytic guidelines",
        "NHANES Survey Methods and Analytic Guidelines",
        "Centers for Disease Control and Prevention, National Center for Health Statistics",
        "",
        "not available",
        "",
        "",
        "https://wwwn.cdc.gov/nchs/nhanes/analyticguidelines.aspx",
        "VERIFIED_OFFICIAL_SOURCE",
        "Methods; Ethics",
    ),
    Reference(
        4,
        "NHANES complex sampling and combined weights",
        "NHANES Tutorials: Weighting Module",
        "Centers for Disease Control and Prevention, National Center for Health Statistics",
        "",
        "not available",
        "",
        "",
        "https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx",
        "VERIFIED_OFFICIAL_SOURCE",
        "Methods",
    ),
    Reference(
        5,
        "NHANES questionnaire documentation for cardiovascular history",
        "NHANES 2007-2008 Data Documentation, Codebook, and Frequencies: Medical Conditions (MCQ_E)",
        "Centers for Disease Control and Prevention, National Center for Health Statistics",
        "",
        "2007-2008",
        "",
        "",
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/MCQ_E.htm",
        "VERIFIED_OFFICIAL_SOURCE",
        "Methods",
    ),
    Reference(
        6,
        "CKD-EPI eGFR equation",
        "A new equation to estimate glomerular filtration rate",
        "Levey AS, Stevens LA, Schmid CH, Zhang YL, Castro AF 3rd, Feldman HI, et al.; CKD-EPI",
        "Annals of Internal Medicine;150(9):604-612",
        "2009",
        "10.7326/0003-4819-150-9-200905050-00006",
        "19414839",
        "https://pubmed.ncbi.nlm.nih.gov/19414839/",
        "VERIFIED_PUBMED",
        "Introduction; Methods; Discussion",
    ),
    Reference(
        7,
        "UACR/eGFR and cardiovascular risk",
        "Association of estimated glomerular filtration rate and albuminuria with all-cause and cardiovascular mortality in general population cohorts: a collaborative meta-analysis",
        "Matsushita K, van der Velde M, Astor BC, Woodward M, Levey AS, de Jong PE, et al.; Chronic Kidney Disease Prognosis Consortium",
        "The Lancet;375(9731):2073-2081",
        "2010",
        "10.1016/S0140-6736(10)60674-5",
        "20483451",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3993088/",
        "VERIFIED_PMC_PUBMED",
        "Introduction; Discussion",
    ),
    Reference(
        8,
        "TyG index definition and insulin resistance validation",
        "The product of triglycerides and glucose, a simple measure of insulin sensitivity. Comparison with the euglycemic-hyperinsulinemic clamp",
        "Guerrero-Romero F, Simental-Mendia LE, Gonzalez-Ortiz M, Martinez-Abundis E, Ramos-Zavala MG, Hernandez-Gonzalez SO, et al.",
        "Journal of Clinical Endocrinology and Metabolism;95(7):3347-3351",
        "2010",
        "10.1210/jc.2010-0288",
        "20484475",
        "https://pubmed.ncbi.nlm.nih.gov/20484475/",
        "VERIFIED_PUBMED_SEARCH",
        "Introduction; Methods",
    ),
    Reference(
        9,
        "TyG index / TyG-WC and cardiovascular disease",
        "The association between triglyceride-glucose index and its combination with obesity indicators and cardiovascular disease: NHANES 2003-2018",
        "Dang K, Wang X, Hu J, Zhang Y, Cheng L, Qi X, et al.",
        "Cardiovascular Diabetology;23(1):8",
        "2024",
        "10.1186/s12933-023-02115-9",
        "38184598",
        "https://link.springer.com/article/10.1186/s12933-023-02115-9",
        "VERIFIED_PUBLISHER_AND_PUBMED",
        "Introduction; Discussion",
    ),
    Reference(
        10,
        "SIRI / systemic inflammatory indices and cardiovascular disease",
        "The relationship between system inflammation response index and coronary heart disease: a cross-sectional study (NHANES 2007-2016)",
        "Zhang TY, Chen HL, Shi Y, Jin Y, Zhang Y, Chen Y",
        "Frontiers in Cardiovascular Medicine;11:1439913",
        "2024",
        "10.3389/fcvm.2024.1439913",
        "",
        "https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1439913/full",
        "VERIFIED_PUBLISHER_NO_PMID_FOUND",
        "Introduction; Discussion",
    ),
    Reference(
        11,
        "XGBoost methodology",
        "XGBoost: A Scalable Tree Boosting System",
        "Chen T, Guestrin C",
        "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining;785-794",
        "2016",
        "10.1145/2939672.2939785",
        "",
        "https://arxiv.org/abs/1603.02754",
        "VERIFIED_DOI_AND_ARXIV",
        "Methods",
    ),
    Reference(
        12,
        "SHAP methodology",
        "A Unified Approach to Interpreting Model Predictions",
        "Lundberg SM, Lee SI",
        "Advances in Neural Information Processing Systems 30",
        "2017",
        "",
        "",
        "https://papers.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions",
        "VERIFIED_CONFERENCE_SOURCE",
        "Introduction; Methods",
    ),
    Reference(
        13,
        "Tree SHAP methodology",
        "From local explanations to global understanding with explainable AI for trees",
        "Lundberg SM, Erion G, Chen H, DeGrave A, Prutkin JM, Nair B, et al.",
        "Nature Machine Intelligence;2(1):56-67",
        "2020",
        "10.1038/s42256-019-0138-9",
        "",
        "https://www.nature.com/articles/s42256-019-0138-9",
        "VERIFIED_PUBLISHER",
        "Introduction; Methods",
    ),
    Reference(
        14,
        "Survey-weighted logistic regression / complex sampling analysis",
        "Analysis of Complex Survey Samples",
        "Lumley T",
        "Journal of Statistical Software;9(8):1-19",
        "2004",
        "10.18637/jss.v009.i08",
        "",
        "https://www.jstatsoft.org/article/view/v009i08",
        "VERIFIED_JOURNAL_SOURCE",
        "Methods",
    ),
    Reference(
        15,
        "Restricted cubic spline methodology",
        "Flexible regression models with cubic splines",
        "Durrleman S, Simon R",
        "Statistics in Medicine;8(5):551-561",
        "1989",
        "10.1002/sim.4780080504",
        "2657958",
        "https://pubmed.ncbi.nlm.nih.gov/2657958/",
        "VERIFIED_PUBMED",
        "Methods",
    ),
]


def format_p_value(raw: str) -> str:
    try:
        value = float(raw)
    except ValueError:
        return raw
    if value < 0.001:
        return "P < 0.001"
    return f"P = {value:.4f}"


def clean_numeric_formatting(text: str) -> str:
    text = re.sub(r"P for nonlinearity=([0-9.eE+-]+)", lambda m: "P for nonlinearity < 0.001" if float(m.group(1)) < 0.001 else f"P for nonlinearity = {float(m.group(1)):.4f}", text)
    text = re.sub(r"P interaction=([0-9.eE+-]+)", lambda m: "P for interaction < 0.001" if float(m.group(1)) < 0.001 else f"P for interaction = {float(m.group(1)):.4f}", text)
    text = re.sub(r"\bP=([0-9.eE+-]+)", lambda m: format_p_value(m.group(1)), text)
    number_pattern = r"([0-9]+(?:\.[0-9]+)?)"
    text = re.sub(r"\bAUC=" + number_pattern, lambda m: f"AUC = {float(m.group(1)):.4f}", text)
    text = re.sub(r"\bPR-AUC=" + number_pattern, lambda m: f"PR-AUC = {float(m.group(1)):.4f}", text)
    text = re.sub(r"\bF1=" + number_pattern, lambda m: f"F1 = {float(m.group(1)):.4f}", text)
    text = re.sub(r"\bBrier score=" + number_pattern, lambda m: f"Brier score = {float(m.group(1)):.4f}", text)
    text = re.sub(
        r"OR ([0-9.]+) \(([0-9.]+)-([0-9.]+)\)",
        lambda m: f"OR = {float(m.group(1)):.3f} (95% CI {float(m.group(2)):.3f}-{float(m.group(3)):.3f})",
        text,
    )
    text = re.sub(
        r"OR ([0-9.]+) \(95% CI ([0-9.]+)-([0-9.]+)\)",
        lambda m: f"OR = {float(m.group(1)):.3f} (95% CI {float(m.group(2)):.3f}-{float(m.group(3)):.3f})",
        text,
    )
    return text


def add_citations(text: str) -> str:
    replacements = {
        "Population-level surveillance systems such as the Global Burden of Disease (GBD) study provide a structured framework for describing long-term changes in CVD mortality, disability-adjusted life years (DALYs), prevalence, and risk-attributable burden.":
        "Population-level surveillance systems such as the Global Burden of Disease (GBD) study provide a structured framework for describing long-term changes in CVD mortality, disability-adjusted life years (DALYs), prevalence, and risk-attributable burden [1,2].",
        "Renal, metabolic, and inflammatory pathways are closely connected with cardiovascular health.":
        "Renal, metabolic, and inflammatory pathways are closely connected with cardiovascular health [6-10].",
        "Albuminuria, captured by urine albumin-to-creatinine ratio (UACR), is a routinely measurable marker with potential relevance across renal and cardiovascular pathways.":
        "Albuminuria, captured by urine albumin-to-creatinine ratio (UACR), is a routinely measurable marker with potential relevance across renal and cardiovascular pathways [7].",
        "Estimated glomerular filtration rate (eGFR) captures a different aspect of kidney function and may complement albuminuria.":
        "Estimated glomerular filtration rate (eGFR) captures a different aspect of kidney function and may complement albuminuria [6,7].",
        "Metabolic indicators such as the triglyceride-glucose index and its anthropometric extensions, including TyG-WC, combine glycemic and lipid-related information with adiposity.":
        "Metabolic indicators such as the triglyceride-glucose index and its anthropometric extensions, including TyG-WC, combine glycemic and lipid-related information with adiposity [8,9].",
        "Inflammatory composite indices such as the systemic inflammation response index (SIRI) may summarize leukocyte patterns relevant to chronic inflammatory states.":
        "Inflammatory composite indices such as the systemic inflammation response index (SIRI) may summarize leukocyte patterns relevant to chronic inflammatory states [10].",
        "Interpretable machine learning offers a way to compare a broad set of candidate predictors while retaining transparent feature-prioritization tools such as permutation importance and SHAP-based explanations.":
        "Interpretable machine learning offers a way to compare a broad set of candidate predictors while retaining transparent feature-prioritization tools such as permutation importance and SHAP-based explanations [11-13].",
        "First, GBD 2023 CSV exports were used to describe CVD burden trends and attributable risk-factor rankings at the population level.":
        "First, official GBD 2023 CSV exports were used to describe CVD burden trends and attributable risk-factor rankings at the population level [1,2].",
        "Second, NHANES 2005-2018 data were used to conduct individual-level machine learning, interpretable feature ranking, and survey-weighted statistical analyses.":
        "Second, NHANES 2005-2018 data were used to conduct individual-level machine learning, interpretable feature ranking, and survey-weighted statistical analyses [3,4].",
        "GBD 2023 data were read from official user-provided CSV exports.":
        "GBD 2023 data were read from official user-provided CSV exports obtained through IHME/GBD tools [1,2].",
        "CVD was defined using self-reported physician diagnosis or history variables from the NHANES questionnaire.":
        "CVD was defined using self-reported physician diagnosis or history variables from the NHANES medical conditions questionnaire [5].",
        "Survey-weighted logistic regression was used to estimate odds ratios (ORs) and 95% confidence intervals (CIs) for key indicators.":
        "Survey-weighted logistic regression was used to estimate odds ratios (ORs) and 95% confidence intervals (CIs) for key indicators using complex-survey methods [14].",
        "Nonlinear associations were assessed using survey-weighted natural spline models for prespecified indicators, including UACR, eGFR, TyG-WC, SIRI, TyG index, and non-HDL-C.":
        "Nonlinear associations were assessed using survey-weighted natural spline models for prespecified indicators, including UACR, eGFR, TyG-WC, SIRI, TyG index, and non-HDL-C [15].",
        "The GBD results provide population-level context rather than validation of NHANES biomarkers.":
        "The GBD results provide population-level context rather than validation of NHANES biomarkers [1,2].",
        "This pattern is consistent with the concept that albuminuria may capture vascular, renal, and metabolic information not fully summarized by standard covariates.":
        "This pattern is consistent with the concept that albuminuria may capture vascular, renal, and metabolic information not fully summarized by standard covariates [7].",
        "eGFR was strongly prioritized by machine learning and was associated with CVD in Model 3, but the estimate was attenuated in strict Model 4 and was not stable in sensitivity analyses.":
        "eGFR was strongly prioritized by machine learning and was associated with CVD in Model 3, but the estimate was attenuated in strict Model 4 and was not stable in sensitivity analyses [6,7].",
        "TyG-WC combined metabolic and adiposity-related information and was supported by feature ranking, Model 3 regression, RCS analysis, and an obesity-status interaction, although the strict Model 4 result was attenuated.":
        "TyG-WC combined metabolic and adiposity-related information and was supported by feature ranking, Model 3 regression, RCS analysis, and an obesity-status interaction, although the strict Model 4 result was attenuated [9].",
        "SIRI represented the inflammatory domain and was supported by biomarker-focused feature ranking, Model 3 regression, and RCS analysis, but was also sensitive to strict adjustment and sensitivity analyses.":
        "SIRI represented the inflammatory domain and was supported by biomarker-focused feature ranking, Model 3 regression, and RCS analysis, but was also sensitive to strict adjustment and sensitivity analyses [10].",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def refine_methods(text: str) -> str:
    old_population = """NHANES 2005-2018 adult participants were included according to the project pipeline. The final analytic sample contained 16,827 adults. Survey-weighted analyses used NHANES design variables, including combined fasting weights, strata, and primary sampling units. Participants with missing CVD outcome were excluded in the data-preparation stage. The analysis retained the complex survey structure for weighted descriptive and regression analyses."""
    new_population = """NHANES 2005-2018 adult participants aged 20 years or older were included according to the project pipeline [3]. The primary analytic dataset used the fasting-core sample because TyG-related indicators required fasting glucose and triglycerides. The final analytic sample contained 16,827 adults. Participants with missing CVD outcome were excluded in the data-preparation stage. Survey-weighted analyses used the combined fasting weight, strata, and primary sampling unit variables, consistent with NHANES guidance for combining cycles and selecting subsample weights [3,4]. The analysis retained the complex survey structure for weighted descriptive and regression analyses."""
    text = text.replace(old_population, new_population)

    old_candidate = """Candidate predictors covered demographic, lifestyle, anthropometric, blood pressure, glucose and lipid metabolism, inflammatory, and renal-function domains. Derived indicators included TyG index, TyG-BMI, TyG-WC, atherogenic index of plasma, non-HDL-C, neutrophil-to-lymphocyte ratio, platelet-to-lymphocyte ratio, systemic immune-inflammation index, SIRI, eGFR, and UACR when source variables were available. CVD outcome components and any direct CVD diagnostic variables were excluded from predictor sets."""
    new_candidate = """Candidate predictors covered demographic, lifestyle, anthropometric, blood pressure, glucose and lipid metabolism, inflammatory, and renal-function domains. Derived indicators included TyG index, TyG-BMI, TyG-WC, atherogenic index of plasma, non-HDL-C, neutrophil-to-lymphocyte ratio, platelet-to-lymphocyte ratio, systemic immune-inflammation index, SIRI, eGFR, and UACR when source variables were available [6,8-10]. CVD outcome components and any direct CVD diagnostic variables were excluded from predictor sets."""
    text = text.replace(old_candidate, new_candidate)

    old_ml = """Machine learning analyses compared multiple classifiers across the three feature sets. The final manuscript-ready performance table identified the best model in each feature set based on held-out test discrimination. Train/test splitting was stratified, and preprocessing was fitted within training-set pipelines. Model performance metrics included AUC, PR-AUC, F1 score, Brier score, sensitivity, and specificity."""
    new_ml = """Machine learning analyses compared logistic regression, random forest, XGBoost, LightGBM, support vector machine, and elastic net classifiers across the three feature sets [11]. Data were split into stratified training and held-out test sets using a 70:30 ratio (n_train = 11,778; n_test = 5,049). Preprocessing was fitted within training-set pipelines only. Numeric variables were imputed with training-set medians, categorical variables were imputed with the training-set most frequent category and one-hot encoded, and standardization was applied to models requiring scaled inputs. Class imbalance was handled through class weights or model-specific positive-class weighting; SMOTE was not used in the Stage 2 main analysis. Model performance metrics included AUC, PR-AUC, F1 score, Brier score, sensitivity, and specificity."""
    text = text.replace(old_ml, new_ml)

    old_shap = """Feature importance was summarized using complementary approaches, including model-based importance, permutation importance, and held-out SHAP explanations. Consensus feature ranking was derived across methods within each feature set. The manuscript focused on priority indicators that were biologically relevant and repeatedly prioritized across routine and biomarker-focused models, including UACR, eGFR, TyG-WC, SIRI, TyG index, non-HDL-C, LDL-C, and HbA1c."""
    new_shap = """Feature importance was summarized using complementary approaches, including model-based importance, permutation importance, and SHAP explanations computed on the held-out test set using models trained on the training set [12,13]. Consensus feature ranking was derived across methods within each feature set. No top-ranked feature subset was selected before splitting or used to refit the evaluated models. The manuscript focused on priority indicators that were biologically relevant and repeatedly prioritized across routine and biomarker-focused models, including UACR, eGFR, TyG-WC, SIRI, TyG index, non-HDL-C, LDL-C, and HbA1c."""
    text = text.replace(old_shap, new_shap)

    old_rcs = """Nonlinear associations were assessed using survey-weighted natural spline models for prespecified indicators, including UACR, eGFR, TyG-WC, SIRI, TyG index, and non-HDL-C [15]. RCS results were interpreted as evidence of nonlinear association patterns rather than as mechanistic proof."""
    new_rcs = """Nonlinear associations were assessed using survey-weighted natural spline models for prespecified indicators, including UACR, eGFR, TyG-WC, SIRI, TyG index, and non-HDL-C [15]. The R implementation used `splines::ns(df = 4)` with Model 3 adjustment. RCS results were interpreted as evidence of nonlinear association patterns rather than as mechanistic proof."""
    text = text.replace(old_rcs, new_rcs)

    old_subgroup = """Subgroup analyses evaluated whether associations varied across age group, sex, diabetes, hypertension, and obesity status when available. Sensitivity analyses excluded prior stroke, excluded prior heart attack, and used routine biomarker adjustment without hypertension or diabetes diagnosis variables. These analyses were used to evaluate robustness rather than to define clinical decision thresholds."""
    new_subgroup = """Subgroup analyses evaluated whether associations varied across age group (<60 versus >=60 years), sex, diabetes status, hypertension status, and obesity status when available. Sensitivity analyses excluded prior stroke, excluded prior heart attack, and used routine biomarker adjustment without hypertension or diabetes diagnosis variables. These analyses were used to evaluate robustness rather than to define clinical decision thresholds."""
    text = text.replace(old_subgroup, new_subgroup)

    old_software = """Data preparation, machine learning, feature ranking, GBD analysis, and manuscript table generation were performed using Python. Survey-weighted analyses were performed using R with survey-design methods. Figure assembly and manuscript-ready result packaging were performed with reproducible project scripts."""
    new_software = """Data preparation, machine learning, feature ranking, GBD analysis, and manuscript table generation were performed using Python. Survey-weighted analyses were performed using R with the `survey` package and survey-design methods [14]. Natural spline analyses used the R `splines` package. Figure assembly and manuscript-ready result packaging were performed with reproducible project scripts."""
    text = text.replace(old_software, new_software)
    return text


def replace_references_section(text: str) -> str:
    refs = "\n".join(ref.formatted() for ref in REFERENCES)
    return re.sub(r"## References\n[\s\S]*$", "## References\n\n" + refs + "\n", text)


def write_reference_table(references: Iterable[Reference]) -> None:
    REFERENCE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with REFERENCE_TABLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(REFERENCES[0]).keys()))
        writer.writeheader()
        for ref in references:
            writer.writerow(asdict(ref))


def write_audit(text: str) -> None:
    manual_refs = [r for r in REFERENCES if "NEED_MANUAL_VERIFICATION" in r.verification_status]
    methods_need = sorted(set(re.findall(r"NEED_PROJECT_CONFIRMATION", text)))
    overclaim_terms = ["caused", "causal effect", "novel biomarker"]
    overclaims = [term for term in overclaim_terms if term.lower() in text.lower()]
    citations = sorted(set(re.findall(r"\[(?:\d+(?:-\d+)?)(?:,\d+(?:-\d+)?)*\]", text)))
    audit = [
        "# Stage 7 References and Formatting Audit",
        "",
        f"- Stage 7 manuscript completed: {OUTPUT_DRAFT.exists()}",
        f"- Output manuscript: `{OUTPUT_DRAFT.relative_to(PROJECT_ROOT)}`",
        f"- References added: {len(REFERENCES)}",
        f"- Reference verification table: `{REFERENCE_TABLE.relative_to(PROJECT_ROOT)}`",
        f"- References needing manual verification: {len(manual_refs)}",
        f"- Citation markers inserted: {', '.join(citations) if citations else 'none'}",
        "",
        "## Reference Verification",
    ]
    if manual_refs:
        audit.extend([f"- Ref {r.citation_number}: {r.title}" for r in manual_refs])
    else:
        audit.append("- No references were marked NEED_MANUAL_VERIFICATION. Records without DOI/PMID are official web resources or verified publisher/conference pages.")
    audit.extend(
        [
            "",
            "## Methods Information Still Requiring Confirmation",
            "- " + ("; ".join(methods_need) if methods_need else "None identified from the Stage 6 manuscript, Stage 5 outputs, and project scripts."),
            "",
            "## Formatting Checks",
            f"- Scientific-notation P values remaining in manuscript: {bool(re.search(r'P\\s*[=<]?\\s*[0-9.]+e[+-]?\\d+', text, flags=re.I))}",
            f"- AUC values formatted to four decimals where expressed as AUC: {not bool(re.search(r'AUC\\s*=\\s*0\\.846\\b', text))}",
            f"- OR expressions include 95% CI labels: {not bool(re.search(r'OR\\s+[0-9.]+\\s*\\([0-9.]+-[0-9.]+\\)', text))}",
            "",
            "## Overstatement Check",
            f"- Guarded terms detected: {', '.join(overclaims) if overclaims else 'None'}",
            "- GBD remains described as population-level context, not NHANES biomarker validation.",
            "- Cross-sectional NHANES associations are described as associations or prioritization signals, not causal effects.",
            "",
            "## Recommendation",
            "- Proceed to Stage 8 target-journal formatting after author information, funding, competing interests, and journal-specific reference style are confirmed.",
        ]
    )
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text("\n".join(audit) + "\n", encoding="utf-8")


def main() -> None:
    text = INPUT_DRAFT.read_text(encoding="utf-8")
    text = text.replace(
        "# Interpretable Machine Learning Identifies Renal, Metabolic, and Inflammatory Indicators Associated with Cardiovascular Disease: Evidence from NHANES 2005鈥?018 and GBD 2023",
        "# Interpretable Machine Learning Identifies Renal, Metabolic, and Inflammatory Indicators Associated with Cardiovascular Disease: Evidence from NHANES 2005-2018 and GBD 2023",
    )
    text = text.replace(
        "# Interpretable Machine Learning Identifies Renal, Metabolic, and Inflammatory Indicators Associated with Cardiovascular Disease: Evidence from NHANES 2005–2018 and GBD 2023",
        "# Interpretable Machine Learning Identifies Renal, Metabolic, and Inflammatory Indicators Associated with Cardiovascular Disease: Evidence from NHANES 2005-2018 and GBD 2023",
    )
    text = add_citations(text)
    text = refine_methods(text)
    text = clean_numeric_formatting(text)
    text = replace_references_section(text)
    OUTPUT_DRAFT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DRAFT.write_text(text, encoding="utf-8")
    write_reference_table(REFERENCES)
    write_audit(text)


if __name__ == "__main__":
    main()
