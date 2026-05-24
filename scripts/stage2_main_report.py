import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402


WATCH_FEATURES = ["egfr", "uacr", "siri", "tyg_wc", "nlr", "non_hdl_cholesterol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Stage 2 main analysis diagnostics and concise result summary.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--fix-note", default="")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def best_models(performance: pd.DataFrame) -> pd.DataFrame:
    return performance.sort_values(["feature_set", "auc"], ascending=[True, False]).groupby("feature_set").head(1)


def top10_by_feature_set(consensus: pd.DataFrame) -> dict:
    return {
        feature_set: group.sort_values("consensus_rank").head(10)["feature"].tolist()
        for feature_set, group in consensus.groupby("feature_set")
    }


def stability_lines(top10: dict) -> list[str]:
    lines = []
    for feature in WATCH_FEATURES:
        present = [name for name, features in top10.items() if feature in features]
        lines.append(f"- `{feature}`: top 10 in {len(present)}/3 feature sets" + (f" ({', '.join(present)})" if present else ""))
    return lines


def write_outputs(root: Path, args: argparse.Namespace) -> None:
    suffix = f"{args.start_year}_{args.end_year}"
    diagnostics = root / "outputs/diagnostics"
    tables = root / "outputs/tables"
    manuscript = root / "outputs/manuscript_results"
    figures = root / "outputs/figures"
    manuscript.mkdir(parents=True, exist_ok=True)

    flow = read_csv(diagnostics / f"sample_size_flow_{suffix}.csv")
    distribution = read_csv(diagnostics / f"cvd_outcome_distribution_{suffix}.csv")
    performance = read_csv(tables / f"model_performance_{suffix}_by_featureset.csv")
    consensus = read_csv(tables / f"feature_ranking_consensus_{suffix}_by_featureset.csv")
    feature_map = read_csv(diagnostics / f"predictor_feature_list_{suffix}.csv")

    # Keep required public aliases current.
    shutil.copyfile(tables / f"model_performance_{suffix}_by_featureset.csv", tables / "table2_model_performance.csv")
    shutil.copyfile(tables / f"feature_ranking_consensus_{suffix}_by_featureset.csv", tables / "table3_feature_ranking.csv")

    final_n = int(flow.iloc[-1]["remaining_n"])
    cvd_cases = int(distribution.loc[distribution["label"].eq("CVD"), "n"].sum())
    non_cvd = int(distribution.loc[distribution["label"].eq("Non-CVD"), "n"].sum())
    best = best_models(performance)
    top10 = top10_by_feature_set(consensus)

    full_auc = float(best.loc[best["feature_set"].eq("full_clinical_model"), "auc"].iloc[0])
    bio_auc = float(best.loc[best["feature_set"].eq("biomarker_focused_model"), "auc"].iloc[0])
    auc_diff = full_auc - bio_auc
    max_auc = float(performance["auc"].max())
    leakage_included = feature_map.loc[
        feature_map["outcome_or_potential_leakage"].eq(True)
        & (
            feature_map["in_full_clinical_model"].eq(True)
            | feature_map["in_routine_indicator_model"].eq(True)
            | feature_map["in_biomarker_focused_model"].eq(True)
        )
    ]
    rscript_available = shutil.which("Rscript") is not None
    gbd_files = list((root / "data/raw/gbd").glob("*.csv"))

    report_lines = [
        "# Stage 2 Main Analysis Report",
        "",
        "No manuscript text and no fabricated results were generated. This report summarizes completed real-data outputs.",
        "",
        "## Run Scope",
        "",
        f"- NHANES years: {args.start_year}-{args.end_year}",
        "- Analysis path: fasting-core sample with three Stage 1.5 feature sets",
        "- ML input: `data/processed/nhanes_filtered_pre_imputation.csv`; imputer/scaler are fit inside training-set pipelines.",
        "",
        "## Leakage Check",
        "",
        f"- Leakage variables included in final feature sets: {len(leakage_included)}",
        "- CVD outcome uses only `chf_code`, `chd_code`, `angina_code`, `heart_attack_code`, and `stroke_code`.",
        "- `cvd`, `cvd_*`, and direct CVD history fields are excluded from predictors.",
        "",
        "## Sample",
        "",
        f"- Final included n: {final_n}",
        f"- CVD cases: {cvd_cases}",
        f"- Non-CVD participants: {non_cvd}",
        "",
        "## Best Model By Feature Set",
        "",
    ]
    for row in best.itertuples(index=False):
        report_lines.append(
            f"- `{row.feature_set}`: {row.model}; AUC={row.auc:.4f}, PR-AUC={row.pr_auc:.4f}, F1={row.f1:.4f}, Brier={row.brier_score:.4f}"
        )
    report_lines.extend(
        [
            "",
            "## AUC Difference",
            "",
            f"- full_clinical_model minus biomarker_focused_model AUC: {auc_diff:.4f}",
            "",
            "## Top 10 Features",
            "",
        ]
    )
    for feature_set in ["full_clinical_model", "routine_indicator_model", "biomarker_focused_model"]:
        if feature_set in top10:
            report_lines.append(f"- `{feature_set}`: {', '.join(top10[feature_set])}")
    report_lines.extend(["", "## Stability Of Priority Indicators", "", *stability_lines(top10), ""])
    report_lines.extend(
        [
            "## AUC/Overfitting Screen",
            "",
            f"- Highest held-out test AUC across models: {max_auc:.4f}",
            "- No model exceeded the pre-specified abnormal screen threshold of AUC > 0.95." if max_auc <= 0.95 else "- WARNING: At least one model exceeded AUC > 0.95 and needs manual overfitting/leakage review.",
            "",
            "## Downstream Recommendation",
            "",
            "- R survey-weighted analysis: " + ("Rscript detected; can proceed." if rscript_available else "Rscript not detected on PATH; install/configure Rscript before running."),
            "- GBD analysis: " + ("GBD CSV files detected; can proceed." if gbd_files else "No GBD CSV files detected in `data/raw/gbd`; add files before running."),
        ]
    )
    if args.fix_note:
        report_lines.extend(["", "## Fixes During Stage 2", "", args.fix_note])
    (diagnostics / "stage2_main_analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8-sig")

    summary_lines = [
        "# Results Summary 2005-2018",
        "",
        f"Final included n: {final_n}. CVD cases: {cvd_cases}; non-CVD: {non_cvd}.",
        "",
        "Best models by feature set:",
    ]
    for row in best.itertuples(index=False):
        summary_lines.append(f"- `{row.feature_set}`: {row.model}, AUC={row.auc:.4f}, PR-AUC={row.pr_auc:.4f}, F1={row.f1:.4f}, Brier={row.brier_score:.4f}")
    summary_lines.extend(["", f"AUC difference, full minus biomarker-focused: {auc_diff:.4f}.", "", "Generated figures:"])
    for figure_name in ["fig2_roc_pr_calibration.png", "fig3_shap_summary.png"]:
        if (figures / figure_name).exists():
            summary_lines.append(f"- `{figure_name}`")
    (manuscript / f"results_summary_{suffix}.md").write_text("\n".join(summary_lines), encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)
    write_outputs(root, args)


if __name__ == "__main__":
    main()
