from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.features import apply_inclusion_exclusion, derive_all_features  # noqa: E402
from cvd_nhanes_gbd.feature_sets import build_predictor_feature_map  # noqa: E402
from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402
from cvd_nhanes_gbd.preprocessing import normalize_missing_values  # noqa: E402


CDC_DATA_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"

MODULE_PREFIXES: Dict[str, List[str]] = {
    "DEMO": ["DEMO_L"],
    "MCQ": ["MCQ_L"],
    "BMX": ["BMX_L"],
    "BPX": ["BPXO_L", "BPX_L"],
    "GLU": ["GLU_L"],
    "GHB": ["GHB_L"],
    "TCHOL": ["TCHOL_L"],
    "HDL": ["HDL_L"],
    "TRIGLY": ["TRIGLY_L"],
    "CBC": ["CBC_L"],
    "ALB_CR": ["ALB_CR_L"],
    "BIOPRO": ["BIOPRO_L"],
    "SMQ": ["SMQ_L"],
    "ALQ": ["ALQ_L"],
    "PAQ": ["PAQ_L"],
    "SLQ": ["SLQ_L"],
    "BPQ": ["BPQ_L"],
    "DIQ": ["DIQ_L"],
}

STANDARD_ALIASES_2021_2023: Dict[str, List[str]] = {
    "seqn": ["SEQN"],
    "age": ["RIDAGEYR"],
    "sex_code": ["RIAGENDR"],
    "race_code": ["RIDRETH3", "RIDRETH1"],
    "education_code": ["DMDEDUC2"],
    "pir": ["INDFMPIR"],
    "mec_weight_2yr": ["WTMEC2YR"],
    "fasting_weight_2yr": ["WTSAF2YR"],
    "strata": ["SDMVSTRA"],
    "psu": ["SDMVPSU"],
    "pregnancy_code": ["RIDEXPRG"],
    "chf_code": ["MCQ160B"],
    "chd_code": ["MCQ160C"],
    "angina_code": ["MCQ160D"],
    "heart_attack_code": ["MCQ160E"],
    "stroke_code": ["MCQ160F"],
    "bmi": ["BMXBMI"],
    "waist": ["BMXWAIST"],
    "sbp_1": ["BPXOSY1", "BPXSY1"],
    "sbp_2": ["BPXOSY2", "BPXSY2"],
    "sbp_3": ["BPXOSY3", "BPXSY3"],
    "sbp_4": ["BPXOSY4", "BPXSY4"],
    "dbp_1": ["BPXODI1", "BPXDI1"],
    "dbp_2": ["BPXODI2", "BPXDI2"],
    "dbp_3": ["BPXODI3", "BPXDI3"],
    "dbp_4": ["BPXODI4", "BPXDI4"],
    "fasting_glucose": ["LBXGLU"],
    "hba1c": ["LBXGH"],
    "total_cholesterol": ["LBXTC"],
    "hdl_cholesterol": ["LBDHDD", "LBDHDL", "LBXHDD"],
    "triglycerides": ["LBXTLG", "LBXTR"],
    "ldl_cholesterol": ["LBDLDL", "LBXLDL", "LBDLDLM", "LBDLDLN"],
    "wbc": ["LBXWBCSI"],
    "neutrophil_abs": ["LBDNENO", "LBXNEUT"],
    "lymphocyte_abs": ["LBDLYMNO", "LBXLYMP"],
    "monocyte_abs": ["LBDMONO", "LBXMONO"],
    "neutrophil_pct": ["LBXNEPCT"],
    "lymphocyte_pct": ["LBXLYPCT"],
    "monocyte_pct": ["LBXMOPCT"],
    "platelet": ["LBXPLTSI"],
    "crp": ["LBXCRP", "LBXHSCRP", "LBXHCRP"],
    "serum_creatinine": ["LBXSCR"],
    "serum_albumin": ["LBXSAL"],
    "uric_acid": ["LBXSUA"],
    "urine_albumin": ["URXUMA"],
    "urine_creatinine": ["URXUCR"],
    "uacr": ["URDACT"],
    "smoking_100_code": ["SMQ020"],
    "smoking_now_code": ["SMQ040"],
    # 2021-2023 ALQ no longer has ALQ101/ALQ110; ALQ111 is a related yes/no alcohol-history item.
    "alcohol_12mo_code": ["ALQ101", "ALQ110", "ALQ111"],
    "alcohol_drinks_per_day": ["ALQ130"],
    # 2021-2023 PAQ no longer directly matches older vigorous/moderate work/recreation yes-no fields.
    "pa_vigorous_work_code": ["PAQ650"],
    "pa_moderate_work_code": ["PAQ665"],
    "pa_vigorous_recreation_code": ["PAQ655"],
    "pa_moderate_recreation_code": ["PAQ670"],
    "sleep_hours": ["SLD010H", "SLD010D", "SLD012"],
    "hypertension_self_code": ["BPQ020"],
    "diabetes_self_code": ["DIQ010"],
}

KEY_EXTERNAL_VARIABLES = [
    "age",
    "sex",
    "race_ethnicity",
    "education",
    "pir",
    "bmi",
    "waist",
    "mean_sbp",
    "mean_dbp",
    "fasting_glucose",
    "hba1c",
    "total_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "ldl_cholesterol",
    "non_hdl_cholesterol",
    "uacr",
    "egfr",
    "tyg_index",
    "tyg_wc",
    "tyg_bmi",
    "nlr",
    "plr",
    "sii",
    "siri",
    "serum_creatinine",
    "crp",
    "physical_activity",
    "alcohol_use",
    "sleep_hours",
    "hypertension",
    "diabetes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NHANES 2021-2023 external temporal validation cohort.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-fasting-core", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def looks_like_xpt(content: bytes) -> bool:
    return content[:32].upper().startswith(b"HEADER RECORD")


def download_xpt(url: str, destination: Path, logger, retries: int = 3) -> Tuple[bool, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=60)
        except requests.RequestException as exc:
            last_error = str(exc)
            logger.warning("Download attempt %s/%s failed for %s: %s", attempt, retries, url, exc)
        else:
            valid = response.status_code == 200 and response.content and looks_like_xpt(response.content)
            if valid:
                destination.write_bytes(response.content)
                logger.info("Downloaded %s", destination)
                return True, ""
            last_error = f"status={response.status_code}; bytes={len(response.content)}; valid_xpt={looks_like_xpt(response.content)}"
            logger.warning("Download attempt %s/%s failed for %s: %s", attempt, retries, url, last_error)
        if attempt < retries:
            time.sleep(1.5 * attempt)
    return False, last_error


def read_xpt(path: Path, logger) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_sas(path, format="xport", encoding="latin1")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    frame.columns = [str(c).upper() for c in frame.columns]
    return frame


def load_module(raw_dir: Path, module: str, download: bool, logger) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
    record = {"module": module, "file": "", "status": "missing", "url": "", "error": ""}
    for prefix in MODULE_PREFIXES[module]:
        filename = f"{prefix}.XPT"
        local_path = raw_dir / filename
        url = f"{CDC_DATA_BASE_URL}/{filename}"
        if not local_path.exists() and download:
            ok, error = download_xpt(url, local_path, logger)
            if not ok:
                record.update({"file": filename, "url": url, "error": error})
                continue
        if local_path.exists():
            frame = read_xpt(local_path, logger)
            if frame is not None and "SEQN" in frame.columns:
                record.update({"file": filename, "status": "loaded", "url": url})
                return frame, record
            record.update({"file": filename, "status": "unreadable", "url": url, "error": "missing SEQN or unreadable XPT"})
    logger.warning("Module %s unavailable for NHANES 2021-2023.", module)
    return None, record


def merge_modules(raw_dir: Path, download: bool, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    demo, demo_record = load_module(raw_dir, "DEMO", download, logger)
    records = [demo_record]
    if demo is None:
        raise RuntimeError("DEMO_L is required but could not be loaded.")
    merged = demo.copy()
    merged["cycle"] = "2021-2023"
    merged["cycle_start"] = 2021
    merged["cycle_end"] = 2023
    merged["cycle_mid_year"] = 2022.0
    for module in MODULE_PREFIXES:
        if module == "DEMO":
            continue
        frame, record = load_module(raw_dir, module, download, logger)
        records.append(record)
        if frame is None:
            continue
        duplicate_cols = [c for c in frame.columns if c != "SEQN" and c in merged.columns]
        if duplicate_cols:
            logger.info("Dropping duplicate columns from %s: %s", module, ", ".join(duplicate_cols))
            frame = frame.drop(columns=duplicate_cols)
        merged = merged.merge(frame, on="SEQN", how="left")
        logger.info("Merged %s; rows=%s columns=%s", module, merged.shape[0], merged.shape[1])
    return merged, pd.DataFrame(records)


def standardize_external_columns(raw: pd.DataFrame, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    df.columns = [str(c).upper() if str(c).isupper() else str(c) for c in df.columns]
    out = pd.DataFrame(index=df.index)
    rows = []
    for standard_name, aliases in STANDARD_ALIASES_2021_2023.items():
        present = [alias for alias in aliases if alias in df.columns]
        if present:
            series = df[present[0]].copy()
            for alias in present[1:]:
                series = series.combine_first(df[alias])
            out[standard_name] = series
            source_found = ";".join(present)
        else:
            logger.warning("No 2021-2023 source variable found for %s", standard_name)
            out[standard_name] = pd.NA
            source_found = ""
        rows.append(
            {
                "standard_name": standard_name,
                "candidate_source_variables": ";".join(aliases),
                "source_found": source_found,
                "source_available": bool(source_found),
            }
        )
    for col in ["cycle", "cycle_start", "cycle_end", "cycle_mid_year"]:
        out[col] = raw[col].values
    return out, pd.DataFrame(rows)


def write_external_diagnostics(
    output_dir: Path,
    mapping: pd.DataFrame,
    module_status: pd.DataFrame,
    standardized: pd.DataFrame,
    final: pd.DataFrame,
    flow: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = mapping.copy()
    for frame_name, frame in [("standardized", standardized), ("final", final)]:
        missing_col = f"{frame_name}_missing_rate"
        mapping[missing_col] = mapping["standard_name"].map(
            lambda c: frame[c].isna().mean() if c in frame.columns else 1.0
        )
    mapping.to_csv(output_dir / "nhanes_2021_2023_variable_mapping.csv", index=False)
    module_status.to_csv(output_dir / "external_module_download_status.csv", index=False)

    missingness = (
        final.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "variable"})
        .sort_values("missing_rate", ascending=False)
    )
    missingness["n_missing"] = missingness["variable"].map(lambda c: int(final[c].isna().sum()))
    missingness["n_total"] = len(final)
    missingness.to_csv(output_dir / "external_missingness_report.csv", index=False)

    flow.to_csv(output_dir / "external_sample_size_flow.csv", index=False)
    dist = final["cvd"].value_counts(dropna=False).rename_axis("cvd").reset_index(name="n")
    dist["label"] = dist["cvd"].map({0.0: "Non-CVD", 1.0: "CVD", 0: "Non-CVD", 1: "CVD"}).fillna("Missing")
    dist.to_csv(output_dir / "external_cvd_distribution.csv", index=False)

    key_status = []
    for variable in KEY_EXTERNAL_VARIABLES:
        key_status.append(
            {
                "variable": variable,
                "available": variable in final.columns,
                "missing_rate": final[variable].isna().mean() if variable in final.columns else 1.0,
            }
        )
    pd.DataFrame(key_status).to_csv(output_dir / "external_key_variable_availability.csv", index=False)
    build_predictor_feature_map(final).to_csv(output_dir / "external_predictor_feature_map.csv", index=False)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    raw_dir = args.raw_dir or (root / "data/raw/nhanes/2021-2023")
    output_dir = root / "outputs/external_validation"
    processed_dir = root / "data/processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("prepare_external_nhanes_2021_2023", root / "outputs/logs/prepare_external_nhanes_2021_2023.log", args.log_level)

    raw, module_status = merge_modules(raw_dir, args.download, logger)
    standardized, mapping = standardize_external_columns(raw, logger)
    derived = derive_all_features(standardized, logger)
    filtered, flow = apply_inclusion_exclusion(derived, logger)

    if args.require_fasting_core:
        before = len(filtered)
        filtered = filtered.loc[filtered["fasting_glucose"].notna() & filtered["triglycerides"].notna()].copy()
        if "combined_fasting_weight" in filtered.columns and filtered["combined_fasting_weight"].notna().any():
            filtered["survey_weight"] = filtered["combined_fasting_weight"]
        flow = pd.concat(
            [
                flow,
                pd.DataFrame(
                    [
                        {
                            "step": "External validation fasting-core restriction: non-missing fasting glucose and triglycerides",
                            "remaining_n": len(filtered),
                            "excluded_n": before - len(filtered),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    filtered = normalize_missing_values(filtered)
    filtered.to_csv(processed_dir / "nhanes_2021_2023_external.csv", index=False)
    write_external_diagnostics(output_dir, mapping, module_status, standardized, filtered, flow)
    logger.info("External NHANES 2021-2023 cohort saved with rows=%s columns=%s", filtered.shape[0], filtered.shape[1])


if __name__ == "__main__":
    main()
