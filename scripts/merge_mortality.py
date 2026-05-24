import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402
from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optionally merge user-provided NHANES linked mortality CSV files into the analytic dataset."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--analysis-input", type=Path, default=None, help="Default: data/processed/nhanes_analysis.csv")
    parser.add_argument("--mortality-dir", type=Path, default=None, help="Directory containing linked mortality CSV files.")
    parser.add_argument("--output", type=Path, default=None, help="Default: data/processed/nhanes_analysis_mortality.csv")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def standardize_mortality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().upper() for c in out.columns]
    rename = {
        "SEQN": "seqn",
        "MORTSTAT": "mortstat",
        "UCOD_LEADING": "ucod_leading",
        "PERMTH_EXM": "permth_exm",
        "PERMTH_INT": "permth_int",
        "ELIGSTAT": "eligstat",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if "seqn" not in out.columns:
        raise ValueError("Mortality CSV must contain SEQN.")
    if "mortstat" in out.columns:
        out["all_cause_mortality"] = np.where(pd.to_numeric(out["mortstat"], errors="coerce").eq(1), 1, 0)
    else:
        out["all_cause_mortality"] = np.nan
    if "ucod_leading" in out.columns:
        cause = pd.to_numeric(out["ucod_leading"], errors="coerce")
        out["cardiovascular_mortality"] = np.where(cause.isin([1, 5]), 1, np.where(cause.notna(), 0, np.nan))
    else:
        out["cardiovascular_mortality"] = np.nan
    keep = ["seqn", "mortstat", "ucod_leading", "permth_exm", "permth_int", "eligstat", "all_cause_mortality", "cardiovascular_mortality"]
    return out[[c for c in keep if c in out.columns]]


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)
    logger = setup_logging("merge_mortality", root / "outputs/logs/merge_mortality.log", args.log_level)

    analysis_input = args.analysis_input or (root / "data/processed/nhanes_analysis.csv")
    mortality_dir = args.mortality_dir or (root / "data/raw/nhanes_mortality")
    output = args.output or (root / "data/processed/nhanes_analysis_mortality.csv")

    files = sorted(mortality_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No mortality CSV files found in {mortality_dir}.")
    mortality = pd.concat([standardize_mortality(pd.read_csv(path)) for path in files], ignore_index=True)
    mortality = mortality.drop_duplicates(subset=["seqn"], keep="first")
    analysis = pd.read_csv(analysis_input)
    merged = analysis.merge(mortality, on="seqn", how="left")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    logger.info("Merged mortality outcomes for %s of %s NHANES participants.", merged["mortstat"].notna().sum(), len(merged))
    logger.info("Saved %s", output)


if __name__ == "__main__":
    main()
