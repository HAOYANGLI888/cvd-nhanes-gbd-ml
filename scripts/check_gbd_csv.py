import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALIASES: Dict[str, List[str]] = {
    "location": ["location", "location_name"],
    "year": ["year", "year_id"],
    "sex": ["sex", "sex_name", "gender", "gender_name"],
    "age": ["age", "age_name"],
    "cause": ["cause", "cause_name"],
    "measure": ["measure", "measure_name"],
    "metric": ["metric", "metric_name"],
    "value": ["value", "val"],
    "lower": ["lower", "lower_value"],
    "upper": ["upper", "upper_value"],
    "rei": ["rei", "rei_name", "risk", "risk_name", "risk_factor", "risk_factor_name"],
}

CORE_FIELDS = ["location", "year", "sex", "age", "cause", "measure", "metric", "value", "lower", "upper"]
RISK_TERMS = [
    "High systolic blood pressure",
    "High LDL cholesterol",
    "High fasting plasma glucose",
    "High body-mass index",
    "Smoking",
    "Dietary risks",
    "Kidney dysfunction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate official IHME/GBD CSV files before Stage 4 analysis.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-dir", type=Path, default=None, help="Default: data/raw/gbd")
    parser.add_argument("--output", type=Path, default=None, help="Default: outputs/diagnostics/gbd_csv_validation_report.md")
    return parser.parse_args()


def clean_name(name: str) -> str:
    name = str(name).strip().lower().replace("\ufeff", "")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    out.columns = [clean_name(c) for c in out.columns]
    rename = {}
    matched = {}
    for canonical, candidates in ALIASES.items():
        for candidate in candidates:
            candidate_clean = clean_name(candidate)
            if candidate_clean in out.columns:
                rename[candidate_clean] = canonical
                matched[canonical] = candidate_clean
                break
    out = out.rename(columns=rename)
    return out, matched


def contains_any(values: Iterable[object], terms: Iterable[str]) -> Dict[str, bool]:
    text = "\n".join(str(v).lower() for v in values if pd.notna(v))
    return {term: term.lower() in text for term in terms}


def norm_set(values: Iterable[object]) -> set:
    return {str(v).strip() for v in values if pd.notna(v) and str(v).strip()}


def format_list(values: Iterable[object], max_items: int = 25) -> str:
    vals = sorted(norm_set(values), key=lambda x: x.lower())
    if not vals:
        return "None"
    if len(vals) > max_items:
        return ", ".join(vals[:max_items]) + f", ... ({len(vals)} total)"
    return ", ".join(vals)


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    input_dir = (args.input_dir or (root / "data/raw/gbd")).resolve()
    output = args.output or (root / "outputs/diagnostics/gbd_csv_validation_report.md")
    output.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(path for path in input_dir.rglob("*.csv") if path.is_file())
    lines: List[str] = [
        "# GBD CSV Validation Report",
        "",
        f"- Input directory: `{input_dir}`",
        f"- CSV files found: {len(csv_files)}",
        "",
    ]

    if not csv_files:
        lines.extend(
            [
                "## Validation Status",
                "",
                "- Overall validation: **FAILED**",
                "- Reason: no CSV files were found in `data/raw/gbd/`.",
                "",
                "Place official IHME/GBD/GHDx CSV exports in `data/raw/gbd/`, then rerun:",
                "",
                "```powershell",
                "python scripts/check_gbd_csv.py",
                "```",
            ]
        )
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"GBD CSV validation FAILED: no CSV files found. Report: {output}")
        return 1

    frames = []
    file_rows = []
    all_missing = {}

    for path in csv_files:
        try:
            preview = pd.read_csv(path, nrows=5)
            full = pd.read_csv(path)
        except Exception as exc:
            rel = str(path.relative_to(input_dir))
            file_rows.append((rel, "READ_ERROR", str(exc), 0, ""))
            all_missing[rel] = CORE_FIELDS + ["rei"]
            continue

        std_preview, matched = standardize_columns(preview)
        std_full, _ = standardize_columns(full)
        missing_core = [field for field in CORE_FIELDS if field not in std_full.columns]
        missing_risk = [] if "rei" in std_full.columns else ["rei"]
        rel = str(path.relative_to(input_dir))
        all_missing[rel] = missing_core + missing_risk
        frames.append(std_full.assign(_source_file=rel))
        file_rows.append(
            (
                rel,
                "OK" if not missing_core else "MISSING_FIELDS",
                ", ".join(missing_core + missing_risk) if (missing_core + missing_risk) else "None",
                len(full),
                ", ".join(std_preview.columns[:20]),
            )
        )

    lines.extend(["## Files", ""])
    lines.append("| file | status | missing fields | rows | first standardized columns |")
    lines.append("|---|---:|---|---:|---|")
    for file_name, status, missing, n_rows, cols in file_rows:
        lines.append(f"| `{file_name}` | {status} | {missing} | {n_rows} | {cols} |")

    if not frames:
        lines.extend(["", "## Validation Status", "", "- Overall validation: **FAILED**", "- Reason: no readable CSV files."])
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"GBD CSV validation FAILED: no readable CSV files. Report: {output}")
        return 1

    data = pd.concat(frames, ignore_index=True, sort=False)
    required_present = all(field in data.columns for field in CORE_FIELDS)
    risk_col_present = "rei" in data.columns and data["rei"].notna().any()

    years = pd.to_numeric(data["year"], errors="coerce") if "year" in data.columns else pd.Series(dtype=float)
    year_min = int(years.min()) if years.notna().any() else None
    year_max = int(years.max()) if years.notna().any() else None

    cause_ok = False
    location_global = False
    location_china = False
    location_us = False
    deaths_ok = False
    dalys_ok = False
    prevalence_ok = False
    number_ok = False
    rate_ok = False
    all_ages_ok = False
    age_std_ok = False
    risk_hits = {term: False for term in RISK_TERMS}

    if "cause" in data.columns:
        cause_ok = data["cause"].astype(str).str.contains("Cardiovascular diseases", case=False, na=False).any()
    if "location" in data.columns:
        loc = data["location"].astype(str)
        location_global = loc.str.fullmatch("Global", case=False, na=False).any()
        location_china = loc.str.fullmatch("China", case=False, na=False).any()
        location_us = loc.str.contains("United States", case=False, na=False).any()
    if "measure" in data.columns:
        measure = data["measure"].astype(str)
        deaths_ok = measure.str.fullmatch("Deaths", case=False, na=False).any()
        dalys_ok = measure.str.contains("DALYs", case=False, na=False).any()
        prevalence_ok = measure.str.fullmatch("Prevalence", case=False, na=False).any()
    if "metric" in data.columns:
        metric = data["metric"].astype(str)
        number_ok = metric.str.fullmatch("Number", case=False, na=False).any()
        rate_ok = metric.str.fullmatch("Rate", case=False, na=False).any()
    if "age" in data.columns:
        age = data["age"].astype(str)
        all_ages_ok = age.str.fullmatch("All ages", case=False, na=False).any()
        age_std_ok = age.str.contains("Age-standardized", case=False, na=False).any()
    if risk_col_present:
        risk_hits = contains_any(data["rei"], RISK_TERMS)

    burden_pass = all(
        [
            required_present,
            cause_ok,
            location_global,
            location_china,
            deaths_ok,
            dalys_ok,
            prevalence_ok,
            number_ok,
            rate_ok,
            all_ages_ok,
            age_std_ok,
        ]
    )
    risk_pass = all(
        [
            required_present,
            cause_ok,
            location_global,
            location_china,
            deaths_ok,
            dalys_ok,
            number_ok,
            rate_ok,
            all_ages_ok,
            age_std_ok,
            risk_col_present,
            any(risk_hits.values()),
        ]
    )
    overall_pass = burden_pass and risk_pass

    lines.extend(
        [
            "",
            "## Content Checks",
            "",
            f"- Core fields present: {required_present}",
            f"- Risk factor field present with data: {risk_col_present}",
            f"- Contains `Cardiovascular diseases`: {cause_ok}",
            f"- Contains Global: {location_global}",
            f"- Contains China: {location_china}",
            f"- Contains United States: {location_us}",
            f"- Contains Deaths: {deaths_ok}",
            f"- Contains DALYs: {dalys_ok}",
            f"- Contains Prevalence: {prevalence_ok}",
            f"- Contains Number: {number_ok}",
            f"- Contains Rate: {rate_ok}",
            f"- Contains All ages: {all_ages_ok}",
            f"- Contains Age-standardized: {age_std_ok}",
            f"- Year range: {year_min} to {year_max}",
            "",
            "## Observed Values",
            "",
            f"- Locations: {format_list(data['location']) if 'location' in data.columns else 'Missing location column'}",
            f"- Measures: {format_list(data['measure']) if 'measure' in data.columns else 'Missing measure column'}",
            f"- Metrics: {format_list(data['metric']) if 'metric' in data.columns else 'Missing metric column'}",
            f"- Sex categories: {format_list(data['sex']) if 'sex' in data.columns else 'Missing sex column'}",
            f"- Age categories: {format_list(data['age']) if 'age' in data.columns else 'Missing age column'}",
            "",
            "## Risk Factor Checks",
            "",
        ]
    )
    for term, ok in risk_hits.items():
        lines.append(f"- {term}: {ok}")

    lines.extend(
        [
            "",
            "## Validation Status",
            "",
            f"- Burden dataset validation: {'PASSED' if burden_pass else 'FAILED'}",
            f"- Risk-factor dataset validation: {'PASSED' if risk_pass else 'FAILED'}",
            f"- Overall validation: **{'PASSED' if overall_pass else 'FAILED'}**",
        ]
    )

    if overall_pass:
        lines.extend(
            [
                "",
                "## Next Command",
                "",
                "```powershell",
                "python gbd_analysis.py --input-dir data/raw/gbd --output-dir outputs",
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Required Fix",
                "",
                "Download or add the missing official GBD CSV exports described in `outputs/diagnostics/gbd_manual_download_guide.md`, then rerun validation.",
            ]
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"GBD CSV validation {'PASSED' if overall_pass else 'FAILED'}. Report: {output}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
