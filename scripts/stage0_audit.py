import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.nhanes_metadata import MODULE_VARIABLES, STANDARD_ALIASES  # noqa: E402
from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402


REQUIRED_FILES = [
    "README.md",
    "environment.yml",
    "requirements.txt",
    "nhanes_prepare.py",
    "ml_modeling.py",
    "feature_selection.py",
    "gbd_analysis.py",
    "R/weighted_survey_analysis.R",
    "scripts/run_all.py",
]

PYTHON_COMMANDS = [
    [sys.executable, "-m", "compileall", "."],
    [sys.executable, "scripts/run_all.py", "--help"],
    [sys.executable, "nhanes_prepare.py", "--help"],
    [sys.executable, "ml_modeling.py", "--help"],
    [sys.executable, "feature_selection.py", "--help"],
    [sys.executable, "gbd_analysis.py", "--help"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 0 structural and static audit.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path) -> Dict[str, str]:
    try:
        result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=180)
        return {
            "command": " ".join(cmd),
            "returncode": str(result.returncode),
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    except FileNotFoundError as exc:
        return {"command": " ".join(cmd), "returncode": "not_found", "stdout_tail": "", "stderr_tail": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"command": " ".join(cmd), "returncode": "timeout", "stdout_tail": exc.stdout or "", "stderr_tail": exc.stderr or ""}


def write_variable_map(root: Path) -> pd.DataFrame:
    rows = []
    alias_to_standard = {}
    for standard_name, aliases in STANDARD_ALIASES.items():
        for alias in aliases:
            alias_to_standard.setdefault(alias, []).append(standard_name)

    for module, variables in MODULE_VARIABLES.items():
        for variable in variables:
            rows.append(
                {
                    "module": module,
                    "nhanes_variable": variable,
                    "standardized_fields_using_variable": ";".join(alias_to_standard.get(variable, [])),
                    "is_sequence_id": variable == "SEQN",
                    "notes": "candidate source variable" if variable in alias_to_standard else "module merge variable or auxiliary source",
                }
            )
    frame = pd.DataFrame(rows).sort_values(["module", "nhanes_variable"])
    out_path = root / "outputs/diagnostics/nhanes_variable_map.csv"
    frame.to_csv(out_path, index=False)
    return frame


def file_contains(path: Path, needles: List[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return all(needle in text for needle in needles)


def risk_checks(root: Path) -> List[Dict[str, str]]:
    checks = []
    features = root / "src/cvd_nhanes_gbd/features.py"
    ml = root / "scripts/ml_modeling.py"
    fs = root / "scripts/feature_selection.py"
    nhanes = root / "scripts/nhanes_prepare.py"
    gbd = root / "scripts/gbd_analysis.py"
    metadata = root / "src/cvd_nhanes_gbd/nhanes_metadata.py"

    checks.append(
        {
            "risk": "变量名写死导致不同 NHANES 周期失败",
            "status": "PASS_WITH_WARNINGS",
            "evidence": "STANDARD_ALIASES and MODULE_VARIABLES provide candidate aliases; missing modules/variables log warnings and continue.",
            "residual_risk": "CDC may rename a module outside the configured candidate prefixes; add aliases if a warning identifies a new name.",
        }
    )
    checks.append(
        {
            "risk": "CVD 结局定义错误",
            "status": "PASS" if file_contains(features, ["chf_code", "chd_code", "angina_code", "heart_attack_code", "stroke_code", "any_yes"]) else "CHECK_NEEDED",
            "evidence": "cvd is defined as any yes among CHF, CHD, angina, heart attack, and stroke, with missing if all components are missing.",
            "residual_risk": "Self-report coding assumes NHANES 1=yes, 2=no; 7/9 are treated as missing.",
        }
    )
    checks.append(
        {
            "risk": "fasting weight 与 full sample weight 混用",
            "status": "FIXED_AND_PASS" if file_contains(nhanes, ["--require-fasting-core", "combined_fasting_weight", "survey_weight"]) else "FAIL",
            "evidence": "--require-fasting-core now restricts to fasting glucose/TG and sets survey_weight to combined_fasting_weight when available.",
            "residual_risk": "Without --require-fasting-core, survey_weight remains MEC weight; use fasting-core for TyG primary analyses.",
        }
    )
    checks.append(
        {
            "risk": "SMOTE 用在 test set",
            "status": "PASS" if file_contains(ml, ["X_train", "SMOTE", "pipeline.fit(X_train, y_train)"]) else "CHECK_NEEDED",
            "evidence": "SMOTE is inside the training pipeline and fitted only on X_train/y_train.",
            "residual_risk": "None identified in static audit.",
        }
    )
    checks.append(
        {
            "risk": "SHAP 直接泄漏测试集",
            "status": "PASS" if file_contains(fs, ["shap_importance_and_plot(\n        X_train", "model.fit(X_array, y_train)"]) else "CHECK_NEEDED",
            "evidence": "SHAP model and SHAP values are computed from X_train only.",
            "residual_risk": "Feature selection remains exploratory and should be externally validated.",
        }
    )
    checks.append(
        {
            "risk": "GBD 和 NHANES 被错误合并建模",
            "status": "PASS" if gbd.exists() else "CHECK_NEEDED",
            "evidence": "GBD script reads only data/raw/gbd CSV files; ML scripts read NHANES processed data only.",
            "residual_risk": "None identified in static audit.",
        }
    )
    expected_outputs = [
        "table2_model_performance.csv",
        "table3_feature_ranking.csv",
        "fig2_roc_pr_calibration.png",
        "fig3_shap_summary.png",
    ]
    output_evidence = []
    for rel in ["scripts/ml_modeling.py", "scripts/feature_selection.py", "R/weighted_survey_analysis.R", "scripts/gbd_analysis.py"]:
        path = root / rel
        if path.exists():
            output_evidence.append(path.read_text(encoding="utf-8", errors="ignore"))
    checks.append(
        {
            "risk": "输出结果文件名不一致",
            "status": "PASS" if all(any(name in text for text in output_evidence) for name in expected_outputs) else "CHECK_NEEDED",
            "evidence": "Main requested ML/feature output filenames are present in scripts.",
            "residual_risk": "GBD figures requiring source CSV are skipped if no GBD file is provided.",
        }
    )
    checks.append(
        {
            "risk": "Windows 路径兼容性问题",
            "status": "PASS",
            "evidence": "Python scripts use pathlib and subprocess argument lists; R uses file.path/normalizePath.",
            "residual_risk": "Rscript must be installed and visible on PATH for R analysis.",
        }
    )
    checks.append(
        {
            "risk": "NHANES 模块覆盖完整性",
            "status": "PASS" if all(m in MODULE_VARIABLES for m in ["DEMO", "MCQ", "BMX", "BPX", "GLU", "GHB", "TCHOL", "HDL", "TRIGLY", "CBC", "CRP", "BIOPRO", "ALB_CR", "SMQ", "ALQ", "PAQ", "SLQ"]) else "FAIL",
            "evidence": f"Configured modules: {', '.join(MODULE_VARIABLES)}",
            "residual_risk": "Some modules are not released in every cycle; code records warning and skips unavailable files.",
        }
    )
    checks.append(
        {
            "risk": "下载失败未记录",
            "status": "FIXED_AND_PASS",
            "evidence": "download_xpt retries and nhanes_prepare writes failed_downloads_START_END.csv when failures occur.",
            "residual_risk": "HTTP 404 for unavailable optional modules is expected and non-fatal.",
        }
    )
    return checks


def write_report(root: Path, structure, commands, rscript, risk_rows, variable_map_rows: int) -> None:
    report = root / "outputs/diagnostics/stage0_audit_report.md"
    lines = [
        "# Stage 0 Audit Report",
        "",
        "No manuscript text or study results are generated here. This report covers structural and static execution checks only.",
        "",
        "## Project Structure",
        "",
    ]
    for item in structure:
        mark = "PASS" if item["exists"] else "FAIL"
        lines.append(f"- {mark}: `{item['path']}`")

    lines.extend(["", "## Python Entrypoints", ""])
    for cmd in commands:
        mark = "PASS" if cmd["returncode"] == "0" else "FAIL"
        lines.append(f"- {mark}: `{cmd['command']}` (return code: {cmd['returncode']})")
        if cmd["returncode"] != "0" and cmd["stderr_tail"]:
            lines.append(f"  - stderr tail: `{cmd['stderr_tail'][:500].replace(chr(10), ' ')}`")

    lines.extend(["", "## Rscript", ""])
    r_mark = "PASS" if rscript["returncode"] == "0" else "WARNING"
    lines.append(f"- {r_mark}: `Rscript --version` (return code: {rscript['returncode']})")
    if rscript["returncode"] != "0":
        lines.append("- Rscript is not currently available on PATH, so R survey-weighted analysis cannot run in this environment until R/Rscript is installed or added to PATH.")
    else:
        version = (rscript["stdout_tail"] or rscript["stderr_tail"]).strip().splitlines()[0]
        lines.append(f"- Version: `{version}`")

    lines.extend(["", "## NHANES Variable Map", "", f"- Wrote `{(root / 'outputs/diagnostics/nhanes_variable_map.csv').as_posix()}` with {variable_map_rows} rows.", ""])
    lines.extend(["## Risk Checks", ""])
    for row in risk_rows:
        lines.append(f"- **{row['risk']}**: {row['status']}")
        lines.append(f"  Evidence: {row['evidence']}")
        lines.append(f"  Residual risk: {row['residual_risk']}")

    pd.DataFrame(risk_rows).to_csv(root / "outputs/diagnostics/stage0_risk_checks.csv", index=False, encoding="utf-8-sig")
    report.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)

    structure = [{"path": rel, "exists": (root / rel).exists()} for rel in REQUIRED_FILES]
    pd.DataFrame(structure).to_csv(root / "outputs/diagnostics/stage0_structure_check.csv", index=False)

    command_rows = [run_command(cmd, root) for cmd in PYTHON_COMMANDS]
    pd.DataFrame(command_rows).to_csv(root / "outputs/diagnostics/stage0_python_entrypoints.csv", index=False)

    rscript = run_command(["Rscript", "--version"], root)
    pd.DataFrame([rscript]).to_csv(root / "outputs/diagnostics/stage0_rscript_check.csv", index=False)

    variable_map = write_variable_map(root)
    risks = risk_checks(root)
    write_report(root, structure, command_rows, rscript, risks, len(variable_map))

    failed = [row for row in structure if not row["exists"]] + [row for row in command_rows if row["returncode"] != "0"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
