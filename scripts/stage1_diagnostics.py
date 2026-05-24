import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Stage 1 diagnostic CSV files for a NHANES trial run.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2018)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)
    diagnostics = root / "outputs/diagnostics"
    tables = root / "outputs/tables"
    processed = root / "data/processed"
    suffix = f"{args.start_year}_{args.end_year}"

    flow = pd.read_csv(tables / "inclusion_exclusion_flow.csv")
    flow.to_csv(diagnostics / f"sample_size_flow_{suffix}.csv", index=False)

    missingness = pd.read_csv(tables / "missingness_report.csv")
    missingness.to_csv(diagnostics / f"missingness_{suffix}.csv", index=False)

    data = pd.read_csv(processed / "nhanes_analysis.csv")
    distribution = (
        data["cvd"]
        .value_counts(dropna=False)
        .rename_axis("cvd")
        .reset_index(name="n")
        .sort_values("cvd")
    )
    total = distribution["n"].sum()
    distribution["percent"] = distribution["n"] / total * 100
    distribution["label"] = distribution["cvd"].map({0.0: "Non-CVD", 1.0: "CVD"}).fillna("Missing")
    distribution.to_csv(diagnostics / f"cvd_outcome_distribution_{suffix}.csv", index=False)

    summary = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "final_n": int(len(data)),
        "cvd_cases": int((data["cvd"] == 1).sum()),
        "non_cvd": int((data["cvd"] == 0).sum()),
        "missing_cvd": int(data["cvd"].isna().sum()),
    }
    pd.DataFrame([summary]).to_csv(diagnostics / f"stage1_summary_{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
