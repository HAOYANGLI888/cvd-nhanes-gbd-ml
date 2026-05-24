import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402
from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a short manuscript-style summary from generated outputs.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def read_csv_if_exists(path: Path):
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)
    logger = setup_logging("summarize_results", root / "outputs/logs/summarize_results.log", args.log_level)
    tables = root / "outputs/tables"
    figures = root / "outputs/figures"
    out_dir = root / "outputs/manuscript_results"

    lines = [
        "# Results Summary",
        "",
        "This file is generated only from completed pipeline outputs. Missing sections indicate that the corresponding analysis has not been run or no eligible input data were available.",
        "",
    ]

    flow = read_csv_if_exists(tables / "inclusion_exclusion_flow.csv")
    if flow is not None and not flow.empty:
        final_n = flow.iloc[-1]["remaining_n"]
        lines.extend(["## NHANES Analytic Sample", "", f"Final analytic sample before high-missingness filtering: **{final_n}** participants.", ""])

    perf = read_csv_if_exists(tables / "table2_model_performance.csv")
    if perf is not None and not perf.empty:
        best = perf.sort_values("auc", ascending=False).iloc[0]
        lines.extend(
            [
                "## Machine Learning Performance",
                "",
                f"Best test-set AUC model: **{best['model']}** (AUC {best['auc']:.3f}, PR-AUC {best['pr_auc']:.3f}, Brier score {best['brier_score']:.3f}).",
                "",
            ]
        )

    ranking = read_csv_if_exists(tables / "table3_feature_ranking.csv")
    if ranking is not None and not ranking.empty:
        top = ranking.head(10)["feature"].tolist()
        lines.extend(["## Consensus Feature Ranking", "", "Top consensus features: " + ", ".join(top) + ".", ""])

    reg = read_csv_if_exists(tables / "table4_weighted_regression.csv")
    if reg is not None and not reg.empty:
        model3 = reg.loc[(reg["model"].astype(str).str.contains("Model 3")) & (reg["contrast"].astype(str).str.contains("per 1-unit"))].head(10)
        lines.extend(["## Weighted Epidemiologic Validation", ""])
        if not model3.empty:
            lines.append("Fully adjusted continuous-indicator estimates are available in `outputs/tables/table4_weighted_regression.csv`.")
        else:
            lines.append("Weighted regression output is available, but no fully adjusted continuous-indicator rows were detected.")
        lines.append("")

    eapc = read_csv_if_exists(tables / "gbd_eapc.csv")
    if eapc is not None and not eapc.empty:
        lines.extend(["## GBD Burden Trends", "", "EAPC estimates for age-standardized CVD burden are available in `outputs/tables/gbd_eapc.csv`.", ""])

    expected_figs = [
        "fig1_gbd_trend.png",
        "fig2_roc_pr_calibration.png",
        "fig3_shap_summary.png",
        "fig4_rcs.png",
        "fig5_subgroup_forest.png",
    ]
    existing = [f for f in expected_figs if (figures / f).exists()]
    if existing:
        lines.extend(["## Generated Figures", "", *[f"- `{name}`" for name in existing], ""])

    summary_path = out_dir / "results_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
