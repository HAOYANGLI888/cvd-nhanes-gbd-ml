import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402
from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete NHANES ML, survey, and GBD workflow.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--impute-method", choices=["iterative", "mice", "median", "none"], default="none")
    parser.add_argument("--require-fasting-core", action="store_true")
    parser.add_argument("--smote", action="store_true")
    parser.add_argument("--skip-nhanes", action="store_true")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-feature-sets", action="store_true")
    parser.add_argument("--legacy-single-feature-ml", action="store_true")
    parser.add_argument("--skip-r", action="store_true")
    parser.add_argument("--skip-gbd", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run_command(cmd, logger, cwd: Path) -> None:
    logger.info("Running: %s", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)
    logger = setup_logging("run_all", root / "outputs/logs/run_all.log", args.log_level)

    if not args.skip_nhanes:
        cmd = [
            sys.executable,
            root / "scripts/nhanes_prepare.py",
            "--project-root",
            root,
            "--start-year",
            args.start_year,
            "--end-year",
            args.end_year,
            "--max-missing",
            args.max_missing,
            "--impute-method",
            args.impute_method,
        ]
        cmd.append("--download" if args.download else "--no-download")
        if args.require_fasting_core:
            cmd.append("--require-fasting-core")
        run_command(cmd, logger, root)
        run_command(
            [
                sys.executable,
                root / "scripts/stage1_diagnostics.py",
                "--project-root",
                root,
                "--start-year",
                args.start_year,
                "--end-year",
                args.end_year,
            ],
            logger,
            root,
        )

    if not args.skip_feature_sets:
        run_command(
            [
                sys.executable,
                root / "scripts/stage1_5_leakage_audit.py",
                "--project-root",
                root,
                "--start-year",
                args.start_year,
                "--end-year",
                args.end_year,
            ],
            logger,
            root,
        )

    if args.legacy_single_feature_ml and not args.skip_ml:
        cmd = [sys.executable, root / "scripts/ml_modeling.py", "--project-root", root]
        if args.smote:
            cmd.append("--smote")
        run_command(cmd, logger, root)

    if args.legacy_single_feature_ml and not args.skip_features:
        run_command([sys.executable, root / "scripts/feature_selection.py", "--project-root", root], logger, root)

    if not args.skip_gbd:
        gbd_files = list((root / "data/raw/gbd").glob("*.csv"))
        if gbd_files:
            run_command([sys.executable, root / "scripts/gbd_analysis.py", "--project-root", root], logger, root)
        else:
            logger.warning("No GBD CSV files found in data/raw/gbd; GBD analysis skipped.")

    if not args.skip_r:
        rscript = shutil.which("Rscript")
        if rscript is None:
            logger.warning("Rscript was not found on PATH; weighted survey analysis skipped.")
        else:
            run_command([rscript, root / "R/weighted_survey_analysis.R", "--project-root", root], logger, root)

    run_command(
        [
            sys.executable,
            root / "scripts/stage2_main_report.py",
            "--project-root",
            root,
            "--start-year",
            args.start_year,
            "--end-year",
            args.end_year,
        ],
        logger,
        root,
    )
    run_command([sys.executable, root / "scripts/summarize_results.py", "--project-root", root], logger, root)
    logger.info("Workflow complete.")


if __name__ == "__main__":
    main()
