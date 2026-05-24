import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.features import apply_inclusion_exclusion, derive_all_features  # noqa: E402
from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402
from cvd_nhanes_gbd.nhanes_io import clear_failed_downloads, get_failed_downloads, load_nhanes_cycles, standardize_columns  # noqa: E402
from cvd_nhanes_gbd.nhanes_metadata import MODULE_SPECS  # noqa: E402
from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402
from cvd_nhanes_gbd.preprocessing import (  # noqa: E402
    drop_high_missingness,
    impute_dataset,
    normalize_missing_values,
    winsorize_continuous,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/read NHANES XPT files and build an analytic CVD dataset.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--start-year", type=int, default=2005, help="First NHANES cycle start year, e.g. 1999 or 2005.")
    parser.add_argument("--end-year", type=int, default=2018, help="Last NHANES cycle end year.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory containing NHANES XPT files by cycle.")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True, help="Download missing XPT files from CDC.")
    parser.add_argument("--modules", type=str, default=",".join(MODULE_SPECS.keys()), help="Comma-separated NHANES modules to load.")
    parser.add_argument("--max-missing", type=float, default=0.30, help="Drop variables with missingness above this fraction.")
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    parser.add_argument("--impute-method", choices=["iterative", "mice", "median", "none"], default="iterative")
    parser.add_argument(
        "--require-fasting-core",
        action="store_true",
        help="Restrict to participants with fasting glucose and triglycerides before missingness filtering; useful for TyG-focused analyses.",
    )
    parser.add_argument("--save-raw", action="store_true", help="Save the merged raw CSV; can be large.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)
    logger = setup_logging("nhanes_prepare", root / "outputs/logs/nhanes_prepare.log", args.log_level)

    raw_dir = args.raw_dir or (root / "data/raw/nhanes")
    processed_dir = root / "data/processed"
    tables_dir = root / "outputs/tables"
    diagnostics_dir = root / "outputs/diagnostics"

    modules = [m.strip().upper() for m in args.modules.split(",") if m.strip()]
    if "DEMO" not in modules:
        modules.insert(0, "DEMO")
    unknown = [m for m in modules if m not in MODULE_SPECS]
    if unknown:
        raise ValueError(f"Unknown module names: {unknown}. Valid modules: {list(MODULE_SPECS)}")

    logger.info("Project root: %s", root)
    logger.info("NHANES cycles: %s-%s; download=%s", args.start_year, args.end_year, args.download)
    clear_failed_downloads()
    raw = load_nhanes_cycles(raw_dir, args.start_year, args.end_year, args.download, logger, modules=modules)
    failed_downloads = get_failed_downloads()
    if not failed_downloads.empty:
        failed_path = diagnostics_dir / f"failed_downloads_{args.start_year}_{args.end_year}.csv"
        failed_downloads.to_csv(failed_path, index=False)
        logger.warning("Saved failed download URLs to %s", failed_path)
    if args.save_raw:
        raw.to_csv(processed_dir / "nhanes_merged_raw.csv", index=False)

    standardized = standardize_columns(raw, logger)
    dictionary = standardized.attrs.get("data_dictionary", pd.DataFrame())
    if not dictionary.empty:
        dictionary.to_csv(tables_dir / "nhanes_data_dictionary.csv", index=False)

    logger.info("Deriving CVD outcome, survey weights, lifestyle variables, and biomarkers.")
    derived = derive_all_features(standardized, logger)
    filtered, flow = apply_inclusion_exclusion(derived, logger)
    if args.require_fasting_core:
        before = len(filtered)
        filtered = filtered.loc[filtered["fasting_glucose"].notna() & filtered["triglycerides"].notna()].copy()
        if "combined_fasting_weight" in filtered.columns and filtered["combined_fasting_weight"].notna().any():
            filtered["survey_weight"] = filtered["combined_fasting_weight"]
            logger.info("Using combined fasting subsample weights after --require-fasting-core.")
        else:
            logger.warning("Fasting weights are unavailable; survey_weight remains MEC weight.")
        flow = pd.concat(
            [
                flow,
                pd.DataFrame(
                    [
                        {
                            "step": "Restrict to non-missing fasting glucose and triglycerides",
                            "remaining_n": len(filtered),
                            "excluded_n": before - len(filtered),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        logger.info("Applied fasting-core restriction; rows=%s", len(filtered))
    flow.to_csv(tables_dir / "inclusion_exclusion_flow.csv", index=False)

    filtered = normalize_missing_values(filtered)
    filtered, missingness = drop_high_missingness(filtered, args.max_missing, logger)
    missingness.to_csv(tables_dir / "missingness_report.csv", index=False)

    filtered, winsor_log = winsorize_continuous(filtered, args.winsor_lower, args.winsor_upper, logger)
    winsor_log.to_csv(tables_dir / "winsorization_log.csv", index=False)
    filtered.to_csv(processed_dir / "nhanes_filtered_pre_imputation.csv", index=False)

    analysis, imputation_log = impute_dataset(filtered, args.impute_method, logger)
    imputation_log.to_csv(tables_dir / "imputation_log.csv", index=False)
    analysis.to_csv(processed_dir / "nhanes_analysis.csv", index=False)
    analysis.to_csv(processed_dir / "nhanes_model.csv", index=False)

    logger.info("Saved analysis dataset: %s", processed_dir / "nhanes_analysis.csv")
    logger.info("Final rows=%s columns=%s", analysis.shape[0], analysis.shape[1])


if __name__ == "__main__":
    main()
