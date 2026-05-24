import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402


ALIASES: Dict[str, List[str]] = {
    "location": ["location", "location_name"],
    "year": ["year", "year_id"],
    "sex": ["sex", "sex_name", "gender", "gender_name"],
    "age": ["age", "age_name"],
    "measure": ["measure", "measure_name"],
    "metric": ["metric", "metric_name"],
    "cause": ["cause", "cause_name"],
    "rei": ["rei", "rei_name", "risk", "risk_name", "risk_factor", "risk_factor_name"],
    "value": ["value", "val"],
    "lower": ["lower", "lower_value"],
    "upper": ["upper", "upper_value"],
}

PREFERRED_RISKS = [
    "High systolic blood pressure",
    "High LDL cholesterol",
    "High fasting plasma glucose",
    "High body-mass index",
    "Smoking",
    "Dietary risks",
    "Kidney dysfunction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze official user-provided GBD CVD burden CSV files.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-dir", type=Path, default=None, help="Default: data/raw/gbd")
    parser.add_argument("--gbd-dir", type=Path, default=None, help="Backward-compatible alias for --input-dir")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: outputs")
    parser.add_argument("--locations", default="Global,China,United States of America,United States")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def clean_name(name: str) -> str:
    name = str(name).strip().lower().replace("\ufeff", "")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def standardize_gbd_columns(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_name(c) for c in out.columns]
    rename = {}
    for canonical, candidates in ALIASES.items():
        for candidate in candidates:
            candidate_clean = clean_name(candidate)
            if candidate_clean in out.columns:
                rename[candidate_clean] = canonical
                break
    out = out.rename(columns=rename)
    required = {"location", "year", "sex", "age", "measure", "metric", "value"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"{source_file} is missing required columns after alias matching: {sorted(missing)}")
    for optional in ["cause", "rei", "lower", "upper"]:
        if optional not in out.columns:
            out[optional] = np.nan
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    for col in ["value", "lower", "upper"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["_source_file"] = source_file
    return out.dropna(subset=["year", "value"]).copy()


def read_gbd_dir(gbd_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    files = sorted(path for path in gbd_dir.rglob("*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No readable GBD CSV files found in {gbd_dir}.")
    frames = []
    used_files = []
    for path in files:
        rel = str(path.relative_to(gbd_dir))
        df = pd.read_csv(path)
        frames.append(standardize_gbd_columns(df, rel))
        used_files.append(rel)
    return pd.concat(frames, ignore_index=True, sort=False), used_files


def filter_cvd(data: pd.DataFrame) -> pd.DataFrame:
    if "cause" not in data.columns or data["cause"].isna().all():
        return data.copy()
    cvd = data.loc[data["cause"].astype(str).str.contains("Cardiovascular diseases", case=False, na=False)].copy()
    if cvd.empty:
        cvd = data.loc[data["cause"].astype(str).str.contains("cardiovascular", case=False, na=False)].copy()
    return cvd


def normalize_location(value: object) -> str:
    text = str(value)
    if text.lower() in {"united states", "united states of america", "usa"}:
        return "United States"
    return text


def select_locations(data: pd.DataFrame, locations: List[str]) -> pd.DataFrame:
    if not locations:
        return data.copy()
    wanted = {normalize_location(x).lower() for x in locations}
    loc_norm = data["location"].map(normalize_location)
    out = data.loc[loc_norm.str.lower().isin(wanted)].copy()
    out["location"] = loc_norm.loc[out.index]
    return out


def canonical_measure(value: object) -> str:
    text = str(value)
    low = text.lower()
    if "daly" in low:
        return "DALYs"
    if low == "deaths" or "death" in low:
        return "Deaths"
    if "prevalence" in low:
        return "Prevalence"
    if "yld" in low:
        return "YLDs"
    return text


def prep_data(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["location"] = out["location"].map(normalize_location)
    out["measure"] = out["measure"].map(canonical_measure)
    out["metric"] = out["metric"].astype(str).str.strip()
    out["age"] = out["age"].astype(str).str.strip()
    out["sex"] = out["sex"].astype(str).str.strip()
    out["cause"] = out["cause"].astype(str).str.strip()
    out["rei"] = out["rei"].where(out["rei"].notna(), np.nan)
    return out


def aggregate_values(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["value", "lower", "upper"])

    def summarize(group: pd.DataFrame) -> pd.Series:
        metric = str(group["metric"].iloc[0]).lower() if "metric" in group else ""
        if "rate" in metric or "percent" in metric:
            value = group["value"].mean()
            lower = group["lower"].mean() if "lower" in group else np.nan
            upper = group["upper"].mean() if "upper" in group else np.nan
        else:
            value = group["value"].sum()
            lower = group["lower"].sum(min_count=1) if "lower" in group else np.nan
            upper = group["upper"].sum(min_count=1) if "upper" in group else np.nan
        return pd.Series({"value": value, "lower": lower, "upper": upper})

    return df.groupby(group_cols, dropna=False).apply(summarize).reset_index()


def t_p_value(t_value: float, df: int) -> float:
    if not math.isfinite(t_value):
        return np.nan
    try:
        from scipy import stats

        return float(2 * stats.t.sf(abs(t_value), df))
    except Exception:
        # Normal approximation fallback if scipy is unavailable.
        return float(math.erfc(abs(t_value) / math.sqrt(2)))


def compute_eapc(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    records = []
    for keys, group in df.groupby(group_cols, dropna=False):
        sub = group.dropna(subset=["year", "value"]).loc[group["value"] > 0].sort_values("year")
        if sub["year"].nunique() < 3:
            continue
        x = sub["year"].to_numpy(dtype=float)
        y = np.log(sub["value"].to_numpy(dtype=float))
        beta, intercept = np.polyfit(x, y, 1)
        y_hat = intercept + beta * x
        residual = y - y_hat
        sxx = ((x - x.mean()) ** 2).sum()
        dfree = len(x) - 2
        se = np.sqrt((residual**2).sum() / dfree / sxx) if dfree > 0 and sxx > 0 else np.nan
        eapc = (np.exp(beta) - 1) * 100
        lower = (np.exp(beta - 1.96 * se) - 1) * 100 if not np.isnan(se) else np.nan
        upper = (np.exp(beta + 1.96 * se) - 1) * 100 if not np.isnan(se) else np.nan
        p_value = t_p_value(beta / se, dfree) if not np.isnan(se) and se > 0 else np.nan
        trend = "stable"
        if not np.isnan(p_value) and p_value < 0.05:
            trend = "increasing" if eapc > 0 else "decreasing"
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        record.update(
            {
                "EAPC": eapc,
                "CI_lower": lower,
                "CI_upper": upper,
                "P_value": p_value,
                "trend_direction": trend,
                "start_year": int(sub["year"].min()),
                "end_year": int(sub["year"].max()),
                "n_years": int(sub["year"].nunique()),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def latest_change_summary(df: pd.DataFrame, locations: List[str], measures: List[str]) -> pd.DataFrame:
    rows = []
    source = df.loc[
        df["location"].isin(locations)
        & df["measure"].isin(measures)
        & df["metric"].str.fullmatch("Rate", case=False, na=False)
        & df["age"].str.contains("Age-standardized", case=False, na=False)
        & df["sex"].str.fullmatch("Both", case=False, na=False)
    ].copy()
    for (location, measure), sub in source.groupby(["location", "measure"], dropna=False):
        sub = sub.sort_values("year")
        if sub.empty:
            continue
        first = sub.iloc[0]
        last = sub.iloc[-1]
        pct = ((last["value"] - first["value"]) / first["value"] * 100) if first["value"] else np.nan
        rows.append(
            {
                "location": location,
                "measure": measure,
                "start_year": int(first["year"]),
                "end_year": int(last["year"]),
                "start_value": first["value"],
                "end_value": last["value"],
                "percent_change": pct,
            }
        )
    return pd.DataFrame(rows)


def plot_burden_trend(age_std: pd.DataFrame, figures_dir: Path) -> bool:
    plot_df = age_std.loc[
        age_std["sex"].str.fullmatch("Both", case=False, na=False)
        & age_std["metric"].str.fullmatch("Rate", case=False, na=False)
        & age_std["measure"].isin(["Deaths", "DALYs", "Prevalence"])
    ].copy()
    if plot_df.empty:
        return False
    g = sns.relplot(
        data=plot_df,
        x="year",
        y="value",
        hue="location",
        col="measure",
        kind="line",
        facet_kws={"sharey": False},
        height=4,
        aspect=1.1,
    )
    g.set_axis_labels("Year", "Age-standardized rate")
    g.fig.suptitle("GBD cardiovascular disease burden trends", y=1.04)
    g.fig.tight_layout()
    g.fig.savefig(figures_dir / "fig1_gbd_cvd_burden_trend.png", dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    return True


def plot_china_global_comparison(age_std: pd.DataFrame, figures_dir: Path) -> bool:
    plot_df = age_std.loc[
        age_std["location"].isin(["Global", "China"])
        & age_std["sex"].str.fullmatch("Both", case=False, na=False)
        & age_std["metric"].str.fullmatch("Rate", case=False, na=False)
        & age_std["measure"].isin(["Deaths", "DALYs", "Prevalence"])
    ].copy()
    if plot_df.empty:
        return False
    g = sns.relplot(
        data=plot_df,
        x="year",
        y="value",
        hue="location",
        col="measure",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=4,
        aspect=1.05,
    )
    g.set_axis_labels("Year", "Age-standardized rate")
    g.fig.suptitle("China versus Global CVD burden", y=1.04)
    g.fig.tight_layout()
    g.fig.savefig(figures_dir / "fig1_gbd_china_global_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    return True


def create_risk_ranking(data: pd.DataFrame) -> pd.DataFrame:
    risk = data.loc[data["rei"].notna() & data["rei"].astype(str).str.strip().ne("")].copy()
    if risk.empty:
        return pd.DataFrame()
    risk = risk.loc[
        risk["location"].isin(["Global", "China"])
        & risk["sex"].str.fullmatch("Both", case=False, na=False)
        & risk["measure"].isin(["Deaths", "DALYs"])
    ].copy()
    if risk.empty:
        return pd.DataFrame()
    latest = int(risk["year"].max())
    risk = risk.loc[risk["year"].eq(latest)].copy()
    ranking = aggregate_values(risk, ["location", "year", "measure", "metric", "age", "sex", "rei"])
    ranking["rank"] = ranking.groupby(["location", "measure", "metric", "age", "sex"])["value"].rank(
        method="dense", ascending=False
    )
    return ranking.sort_values(["location", "measure", "metric", "age", "rank", "rei"]).copy()


def plot_risk_ranking(ranking: pd.DataFrame, figures_dir: Path) -> bool:
    if ranking.empty:
        return False
    plot_df = ranking.loc[
        ranking["measure"].eq("DALYs")
        & ranking["metric"].str.fullmatch("Number", case=False, na=False)
        & ranking["age"].str.fullmatch("All ages", case=False, na=False)
        & ranking["sex"].str.fullmatch("Both", case=False, na=False)
    ].copy()
    if plot_df.empty:
        plot_df = ranking.loc[
            ranking["metric"].str.fullmatch("Number", case=False, na=False)
            & ranking["age"].str.fullmatch("All ages", case=False, na=False)
        ].copy()
    if plot_df.empty:
        return False
    plot_df = plot_df.sort_values(["location", "value"], ascending=[True, True])
    height = max(5, 0.35 * len(plot_df))
    plt.figure(figsize=(9, height))
    sns.barplot(data=plot_df, x="value", y="rei", hue="location")
    plt.xlabel("Attributable burden in latest year")
    plt.ylabel("")
    plt.title("CVD attributable risk-factor burden")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig1_gbd_risk_factor_ranking.png", dpi=300)
    plt.close()
    return True


def fmt_num(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.{digits}f} million"
    if abs(value) >= 1_000:
        return f"{value:,.{digits}f}"
    return f"{value:.{digits}f}"


def format_range(values: Iterable[object], max_items: int = 20) -> str:
    vals = sorted({str(v) for v in values if pd.notna(v)}, key=str.lower)
    if not vals:
        return "None"
    if len(vals) > max_items:
        return ", ".join(vals[:max_items]) + f", ... ({len(vals)} total)"
    return ", ".join(vals)


def risk_top_summary(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    key = ranking.loc[
        ranking["measure"].eq("DALYs")
        & ranking["metric"].str.fullmatch("Number", case=False, na=False)
        & ranking["age"].str.fullmatch("All ages", case=False, na=False)
        & ranking["sex"].str.fullmatch("Both", case=False, na=False)
    ].copy()
    if key.empty:
        key = ranking.copy()
    return key.loc[key["rank"].le(5)].sort_values(["location", "rank"])


def write_report(
    path: Path,
    used_files: List[str],
    data: pd.DataFrame,
    trend: pd.DataFrame,
    eapc: pd.DataFrame,
    ranking: pd.DataFrame,
    figure_status: Dict[str, bool],
) -> None:
    year_min = int(data["year"].min()) if not data.empty else None
    year_max = int(data["year"].max()) if not data.empty else None
    trend_focus = latest_change_summary(data, ["Global", "China"], ["Deaths", "DALYs", "Prevalence"])
    risk_focus = risk_top_summary(ranking)
    lines = [
        "# Stage 4 GBD Analysis Report",
        "",
        "## Data Sources",
        "",
    ]
    lines.extend([f"- `{f}`" for f in used_files])
    lines.extend(
        [
            "",
            "## Data Coverage",
            "",
            f"- Year range: {year_min} to {year_max}",
            f"- Locations: {format_range(data['location'])}",
            f"- Measures: {format_range(data['measure'])}",
            f"- Metrics: {format_range(data['metric'])}",
            f"- Ages: {format_range(data['age'])}",
            f"- Sex: {format_range(data['sex'])}",
            f"- Risk factors: {format_range(data['rei'])}",
            "",
            "## Global and China CVD Burden Trends",
            "",
        ]
    )
    if trend_focus.empty:
        lines.append("- No age-standardized Both-sex rate trend was available for Global/China.")
    else:
        for _, row in trend_focus.sort_values(["location", "measure"]).iterrows():
            direction = "increased" if row["percent_change"] > 0 else "decreased"
            lines.append(
                f"- {row['location']} {row['measure']} age-standardized rate {direction} "
                f"from {fmt_num(row['start_value'])} in {int(row['start_year'])} to "
                f"{fmt_num(row['end_value'])} in {int(row['end_year'])} "
                f"({row['percent_change']:.1f}% change)."
            )
    lines.extend(["", "## Main Attributable Risk Factors", ""])
    if risk_focus.empty:
        lines.append("- No risk-factor ranking could be generated from the available files.")
    else:
        for loc, sub in risk_focus.groupby("location"):
            top = "; ".join([f"{int(r['rank'])}. {r['rei']} ({fmt_num(r['value'])})" for _, r in sub.iterrows()])
            lines.append(f"- {loc}: {top}.")
    lines.extend(
        [
            "",
            "## Output Status",
            "",
            f"- Burden trend table: {not trend.empty}",
            f"- EAPC table: {not eapc.empty}",
            f"- Risk-factor ranking table: {not ranking.empty}",
        ]
    )
    for fig, ok in figure_status.items():
        lines.append(f"- {fig}: {ok}")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- The GBD outputs are suitable as manuscript Result 1/Figure 1 if the author wants a macro-level public-health burden context before the NHANES individual-level analysis.",
            "- GBD and NHANES remain analytically separate; these GBD results should not be interpreted as individual-level prediction evidence.",
            "",
            "## Recommended Main Text Items",
            "",
            "- Table: `outputs/tables/gbd_cvd_burden_trend.csv` or EAPC summary.",
            "- Figure: `outputs/figures/fig1_gbd_cvd_burden_trend.png`.",
            "- Optional Figure: `outputs/figures/fig1_gbd_risk_factor_ranking.png` if risk-factor context is included.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_skeleton(path: Path, data: pd.DataFrame, eapc: pd.DataFrame, ranking: pd.DataFrame) -> None:
    year_min = int(data["year"].min()) if not data.empty else None
    year_max = int(data["year"].max()) if not data.empty else None
    trend_focus = latest_change_summary(data, ["Global", "China", "United States"], ["Deaths", "DALYs", "Prevalence"])
    risk_focus = risk_top_summary(ranking)
    lines = [
        "# GBD Results Skeleton",
        "",
        "## GBD Data Coverage",
        "",
        f"- The GBD analysis used official user-provided CSV exports covering {year_min}-{year_max}.",
        f"- Locations available in the analytic GBD files were: {format_range(data['location'])}.",
        f"- Measures included: {format_range(data['measure'])}; metrics included: {format_range(data['metric'])}.",
        "",
        "## Cardiovascular Disease Burden Trends",
        "",
    ]
    if trend_focus.empty:
        lines.append("- Age-standardized Both-sex rate trends were not available in the provided GBD files.")
    else:
        for _, row in trend_focus.sort_values(["location", "measure"]).iterrows():
            direction = "increased" if row["percent_change"] > 0 else "decreased"
            lines.append(
                f"- In {row['location']}, the age-standardized {row['measure']} rate {direction} "
                f"from {fmt_num(row['start_value'])} in {int(row['start_year'])} to "
                f"{fmt_num(row['end_value'])} in {int(row['end_year'])} "
                f"({row['percent_change']:.1f}% change)."
            )
    lines.extend(["", "## EAPC", ""])
    eapc_focus = eapc.loc[
        eapc["sex"].astype(str).str.fullmatch("Both", case=False, na=False)
        & eapc["measure"].isin(["Deaths", "DALYs", "Prevalence"])
        & eapc["location"].isin(["Global", "China", "United States"])
    ].copy()
    if eapc_focus.empty:
        lines.append("- No EAPC estimates were available.")
    else:
        for _, row in eapc_focus.sort_values(["location", "measure"]).iterrows():
            lines.append(
                f"- {row['location']} {row['measure']}: EAPC={row['EAPC']:.3f}% "
                f"(95% CI {row['CI_lower']:.3f} to {row['CI_upper']:.3f}), "
                f"P={row['P_value']:.3g}; trend={row['trend_direction']}."
            )
    lines.extend(["", "## Attributable Risk Factors", ""])
    if risk_focus.empty:
        lines.append("- Attributable risk-factor ranking was not generated because no risk-factor field was available.")
    else:
        for loc, sub in risk_focus.groupby("location"):
            parts = [f"{int(r['rank'])}. {r['rei']} ({fmt_num(r['value'])})" for _, r in sub.iterrows()]
            lines.append(f"- In {loc}, the top attributable CVD DALY risk factors in the latest year were: " + "; ".join(parts) + ".")
    lines.extend(
        [
            "",
            "## Framing",
            "",
            "- These GBD results describe macro-level disease burden trends and attributable-risk context only. They were not merged with NHANES and were not used for individual-level prediction modeling.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    input_dir = (args.input_dir or args.gbd_dir or (root / "data/raw/gbd")).resolve()
    output_dir = (args.output_dir or (root / "outputs")).resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    diagnostics_dir = output_dir / "diagnostics"
    manuscript_dir = output_dir / "manuscript_results"
    logs_dir = output_dir / "logs"
    for path in [tables_dir, figures_dir, diagnostics_dir, manuscript_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("gbd_analysis", logs_dir / "gbd_analysis.log", args.log_level)
    logger.info("Reading GBD CSV files from %s", input_dir)
    data, used_files = read_gbd_dir(input_dir)
    data = prep_data(filter_cvd(data))
    data = select_locations(data, [x.strip() for x in args.locations.split(",") if x.strip()])
    if data.empty:
        raise RuntimeError("GBD data are empty after CVD/location filtering.")

    burden = data.loc[data["rei"].isna() | data["rei"].astype(str).str.strip().eq("")].copy()
    if burden.empty:
        burden = data.loc[~data["measure"].isin(["Deaths", "DALYs"]) | data["rei"].isna()].copy()
    burden_trend = aggregate_values(burden, ["location", "year", "sex", "age", "measure", "metric", "cause"])
    burden_trend = burden_trend.sort_values(["location", "measure", "metric", "age", "sex", "year"])
    burden_trend.to_csv(tables_dir / "gbd_cvd_burden_trend.csv", index=False)

    age_std = burden_trend.loc[
        burden_trend["age"].astype(str).str.contains("Age-standardized", case=False, na=False)
        & burden_trend["metric"].astype(str).str.fullmatch("Rate", case=False, na=False)
    ].copy()
    eapc = compute_eapc(age_std, ["location", "measure", "sex"])
    eapc.to_csv(tables_dir / "gbd_eapc_summary.csv", index=False)

    ranking = create_risk_ranking(data)
    ranking.to_csv(tables_dir / "gbd_risk_factor_ranking.csv", index=False)

    figure_status = {
        "fig1_gbd_cvd_burden_trend.png": plot_burden_trend(age_std, figures_dir),
        "fig1_gbd_china_global_comparison.png": plot_china_global_comparison(age_std, figures_dir),
        "fig1_gbd_risk_factor_ranking.png": plot_risk_ranking(ranking, figures_dir),
    }
    write_report(diagnostics_dir / "stage4_gbd_analysis_report.md", used_files, data, burden_trend, eapc, ranking, figure_status)
    write_skeleton(manuscript_dir / "gbd_results_skeleton_en.md", data, eapc, ranking)
    logger.info("Saved Stage 4 GBD tables, figures, report, and manuscript skeleton.")


if __name__ == "__main__":
    main()
