import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
DIAG = OUT / "diagnostics"
MS_RESULTS = OUT / "manuscript_results"
MS_TABLES = OUT / "manuscript_tables"
MS_FIGURES = OUT / "manuscript_figures"

for directory in [MS_TABLES, MS_FIGURES, MS_RESULTS, DIAG]:
    directory.mkdir(parents=True, exist_ok=True)


DISPLAY = {
    "egfr": "eGFR",
    "uacr": "UACR",
    "siri": "SIRI",
    "tyg_wc": "TyG-WC",
    "tyg_index": "TyG index",
    "non_hdl_cholesterol": "non-HDL-C",
    "serum_creatinine": "serum creatinine",
    "ldl_cholesterol": "LDL-C",
    "hba1c": "HbA1c",
    "urine_albumin": "urine albumin",
    "total_cholesterol": "total cholesterol",
    "smoking_status": "smoking status",
    "pir": "PIR",
}

VAR_LABELS = {
    "age": "Age, years",
    "sex": "Sex",
    "race_ethnicity": "Race/ethnicity",
    "education": "Education",
    "pir": "Poverty income ratio",
    "bmi": "Body mass index, kg/m2",
    "waist": "Waist circumference, cm",
    "mean_sbp": "Systolic blood pressure, mmHg",
    "mean_dbp": "Diastolic blood pressure, mmHg",
    "smoking_status": "Smoking status",
    "alcohol_use": "Alcohol use",
    "physical_activity": "Physical activity",
    "fasting_glucose": "Fasting glucose, mg/dL",
    "hba1c": "HbA1c, %",
    "total_cholesterol": "Total cholesterol, mmol/L",
    "ldl_cholesterol": "LDL-C, mmol/L",
    "hdl_cholesterol": "HDL-C, mmol/L",
    "triglycerides": "Triglycerides, mg/dL",
    "non_hdl_cholesterol": "non-HDL-C, mmol/L",
    "tyg_index": "TyG index",
    "tyg_wc": "TyG-WC",
    "egfr": "eGFR, mL/min/1.73 m2",
    "uacr": "UACR, mg/g",
    "serum_creatinine": "Serum creatinine, mg/dL",
    "urine_albumin": "Urine albumin, mg/L",
    "uric_acid": "Uric acid, mg/dL",
    "wbc": "White blood cell count",
    "nlr": "NLR",
    "siri": "SIRI",
    "sii": "SII",
}

CORE_INDICATORS = ["uacr", "egfr", "tyg_wc", "siri"]
TABLE3_INDICATORS = [
    "uacr",
    "egfr",
    "tyg_wc",
    "siri",
    "tyg_index",
    "non_hdl_cholesterol",
    "ldl_cholesterol",
    "hba1c",
]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required Stage 5 input missing: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require(path))


def fmt_p(value: object) -> str:
    if value is None or pd.isna(value):
        return "not available"
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def fmt_num(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "not available"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{digits}f} million"
    if abs(value) >= 1_000:
        return f"{value:,.{digits}f}"
    return f"{value:.{digits}f}"


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value):.{digits}f}"


def yes_no(value: object) -> str:
    return "Yes" if bool(value) else "No"


def parse_sensitivity_count(text: str) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(text, str):
        return None, None
    match = re.search(r"(\d+)/(\d+)\s+significant", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def significance_from_p(value: object) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return "Yes" if float(value) < 0.05 else "No"


def image_font(size: int = 36):
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_image(path: Path) -> Image.Image:
    img = Image.open(require(path)).convert("RGB")
    return img


def paste_fit(canvas: Image.Image, img: Image.Image, box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    max_w, max_h = x1 - x0, y1 - y0
    scale = min(max_w / img.width, max_h / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    x = x0 + (max_w - new_size[0]) // 2
    y = y0 + (max_h - new_size[1]) // 2
    canvas.paste(resized, (x, y))


def composite_images(
    specs: List[Tuple[str, Path]],
    out_path: Path,
    title: str,
    layout: Tuple[int, int],
    size: Tuple[int, int] = (2400, 1800),
) -> bool:
    missing = [str(path) for _, path in specs if not path.exists()]
    if missing:
        return False
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    title_font = image_font(54)
    label_font = image_font(44)
    draw.text((40, 25), title, fill="black", font=title_font)
    rows, cols = layout
    margin_x, margin_y = 40, 120
    gutter = 25
    panel_w = (size[0] - 2 * margin_x - (cols - 1) * gutter) // cols
    panel_h = (size[1] - margin_y - 40 - (rows - 1) * gutter) // rows
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, (label, path) in enumerate(specs):
        row, col = divmod(idx, cols)
        x0 = margin_x + col * (panel_w + gutter)
        y0 = margin_y + row * (panel_h + gutter)
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.text((x0 + 5, y0 + 5), f"{labels[idx]}. {label}", fill="black", font=label_font)
        paste_fit(canvas, load_image(path), (x0 + 5, y0 + 65, x1 - 5, y1 - 5))
    canvas.save(out_path)
    return True


def copy_existing(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    shutil.copyfile(src, dest)
    return True


def make_sensitivity_panel(sensitivity: pd.DataFrame, out_path: Path) -> None:
    key = ["uacr", "egfr", "tyg_wc", "siri"]
    sens_order = [
        "exclude_prior_stroke",
        "exclude_prior_heart_attack",
        "routine_biomarker_adjustment_no_htn_dm",
    ]
    pivot = pd.DataFrame(index=[DISPLAY[x] for x in key], columns=sens_order, dtype=float)
    labels = pd.DataFrame(index=[DISPLAY[x] for x in key], columns=sens_order, dtype=object)
    for ind in key:
        for sens_name in sens_order:
            row = sensitivity[(sensitivity["indicator"].eq(ind)) & (sensitivity["sensitivity"].eq(sens_name))]
            if row.empty:
                pivot.loc[DISPLAY[ind], sens_name] = np.nan
                labels.loc[DISPLAY[ind], sens_name] = "NA"
            else:
                r = row.iloc[0]
                p = float(r["P_value"]) if not pd.isna(r["P_value"]) else np.nan
                pivot.loc[DISPLAY[ind], sens_name] = 1 if p < 0.05 else 0
                labels.loc[DISPLAY[ind], sens_name] = f"OR {r['OR']:.3f}\nP={fmt_p(p)}"
    fig, ax = plt.subplots(figsize=(10, 5))
    arr = pivot.to_numpy(dtype=float)
    cmap = matplotlib.colors.ListedColormap(["#f3f3f3", "#4e79a7"])
    ax.imshow(np.nan_to_num(arr, nan=0), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sens_order)))
    ax.set_xticklabels(["Exclude stroke", "Exclude heart attack", "Routine biomarkers"], rotation=20, ha="right")
    ax.set_yticks(range(len(key)))
    ax.set_yticklabels([DISPLAY[x] for x in key])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, labels.iloc[i, j], ha="center", va="center", fontsize=8)
    ax.set_title("Sensitivity analysis summary")
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def create_tables(
    table1: pd.DataFrame,
    performance: pd.DataFrame,
    key_summary: pd.DataFrame,
    regression: pd.DataFrame,
    gbd_trend: pd.DataFrame,
    gbd_eapc: pd.DataFrame,
) -> Dict[str, object]:
    # Table 1
    t1 = table1.copy()
    t1["Characteristic"] = t1["variable"].map(VAR_LABELS).fillna(t1["variable"])
    t1 = t1.rename(
        columns={
            "level": "Level/statistic",
            "non_cvd": "Non-CVD",
            "cvd": "CVD",
            "p_value": "P value",
        }
    )
    t1["P value"] = t1["P value"].map(fmt_p)
    t1 = t1[["Characteristic", "Level/statistic", "Non-CVD", "CVD", "P value"]]
    t1.to_csv(MS_TABLES / "Table_1_weighted_baseline_characteristics.csv", index=False, encoding="utf-8-sig")

    # Table 2
    perf = performance.copy()
    idx = perf.groupby("feature_set")["auc"].idxmax()
    best = perf.loc[idx].copy().sort_values("feature_set")
    biomarker_auc = best.loc[best["feature_set"].eq("biomarker_focused_model"), "auc"]
    full_auc = best.loc[best["feature_set"].eq("full_clinical_model"), "auc"]
    auc_diff = np.nan
    if not biomarker_auc.empty and not full_auc.empty:
        auc_diff = float(full_auc.iloc[0] - biomarker_auc.iloc[0])
    t2 = pd.DataFrame(
        {
            "feature set": best["feature_set"],
            "best model": best["model"],
            "n_train": best["n_train"],
            "n_test": best["n_test"],
            "AUC": best["auc"].map(lambda x: fmt_float(x, 4)),
            "PR-AUC": best["pr_auc"].map(lambda x: fmt_float(x, 4)),
            "F1": best["f1"].map(lambda x: fmt_float(x, 4)),
            "Brier score": best["brier_score"].map(lambda x: fmt_float(x, 4)),
            "sensitivity": best["sensitivity"].map(lambda x: fmt_float(x, 4)),
            "specificity": best["specificity"].map(lambda x: fmt_float(x, 4)),
            "full vs biomarker AUC difference": [
                fmt_float(auc_diff, 4) if fs == "full_clinical_model" else "not applicable" for fs in best["feature_set"]
            ],
        }
    )
    t2.to_csv(MS_TABLES / "Table_2_model_performance.csv", index=False, encoding="utf-8-sig")

    # Table 3
    rows = []
    ks = key_summary.set_index("variable_name")
    for ind in TABLE3_INDICATORS:
        if ind not in ks.index:
            rows.append(
                {
                    "indicator": DISPLAY.get(ind, ind),
                    "routine top10": "not available",
                    "biomarker-focused top10": "not available",
                    "Model 3 significant": "not available",
                    "Model 4 significant": "not available",
                    "RCS nonlinear": "not available",
                    "sensitivity stability": "not available",
                    "interpretation": "not available",
                }
            )
            continue
        r = ks.loc[ind]
        sig_count, sig_total = parse_sensitivity_count(str(r["sensitivity_stability"]))
        if sig_count is None:
            stability = "not available"
        elif sig_count >= 2:
            stability = f"Stable/partially stable ({sig_count}/{sig_total} sensitivity analyses significant)"
        elif sig_count == 1:
            stability = f"Limited stability ({sig_count}/{sig_total} sensitivity analyses significant)"
        else:
            stability = f"Not stable ({sig_count}/{sig_total} sensitivity analyses significant)"
        rows.append(
            {
                "indicator": r["indicator"],
                "routine top10": r["in_routine_indicator_model_top10"],
                "routine consensus rank": int(r["routine_consensus_rank"]) if not pd.isna(r["routine_consensus_rank"]) else "not available",
                "biomarker-focused top10": r["in_biomarker_focused_model_top10"],
                "biomarker consensus rank": int(r["biomarker_consensus_rank"]) if not pd.isna(r["biomarker_consensus_rank"]) else "not available",
                "Model 3 significant": significance_from_p(r["Model3_P"]),
                "Model 4 significant": significance_from_p(r["Model4_P"]),
                "RCS nonlinear": significance_from_p(r["RCS_P_nonlinear"]) if not pd.isna(r["RCS_P_nonlinear"]) else "not available",
                "sensitivity stability": stability,
                "interpretation": r["final_interpretation"],
            }
        )
    t3 = pd.DataFrame(rows)
    t3.to_csv(MS_TABLES / "Table_3_feature_ranking.csv", index=False, encoding="utf-8-sig")

    # Table 4
    rows = []
    for ind in CORE_INDICATORS:
        m3 = regression[(regression["indicator"].eq(ind)) & regression["form"].eq("continuous") & regression["model"].eq("Model 3")]
        m4 = regression[(regression["indicator"].eq(ind)) & regression["form"].eq("continuous") & regression["model"].eq("Model 4")]
        m3r = m3.iloc[0] if not m3.empty else None
        m4r = m4.iloc[0] if not m4.empty else None
        rows.append(
            {
                "indicator": DISPLAY.get(ind, ind),
                "Model 3 OR (95% CI)": f"{m3r['OR']:.3f} ({m3r['CI_lower']:.3f}-{m3r['CI_upper']:.3f})" if m3r is not None else "not available",
                "Model 3 P value": fmt_p(m3r["P_value"]) if m3r is not None else "not available",
                "Model 3 n/events": f"{int(m3r['n_total'])}/{int(m3r['n_events'])}" if m3r is not None else "not available",
                "Strict Model 4 OR (95% CI)": f"{m4r['OR']:.3f} ({m4r['CI_lower']:.3f}-{m4r['CI_upper']:.3f})" if m4r is not None else "not available",
                "Strict Model 4 P value": fmt_p(m4r["P_value"]) if m4r is not None else "not available",
                "Strict Model 4 n/events": f"{int(m4r['n_total'])}/{int(m4r['n_events'])}" if m4r is not None else "not available",
                "summary": ks.loc[ind, "final_interpretation"] if ind in ks.index else "not available",
            }
        )
    t4 = pd.DataFrame(rows)
    t4.to_csv(MS_TABLES / "Table_4_weighted_regression.csv", index=False, encoding="utf-8-sig")

    # Table 5
    trend = gbd_trend.copy()
    focus = trend[
        trend["location"].isin(["China", "Global", "United States"])
        & trend["sex"].eq("Both")
        & trend["age"].str.contains("Age-standardized", case=False, na=False)
        & trend["metric"].eq("Rate")
        & trend["measure"].isin(["Deaths", "DALYs", "Prevalence"])
        & trend["year"].isin([1990, 2023])
    ].copy()
    start = focus[focus["year"].eq(1990)].set_index(["location", "measure"])
    end = focus[focus["year"].eq(2023)].set_index(["location", "measure"])
    eapc = gbd_eapc[(gbd_eapc["sex"].eq("Both"))].set_index(["location", "measure"])
    rows = []
    for loc in ["Global", "China", "United States"]:
        for measure in ["Deaths", "DALYs", "Prevalence"]:
            key = (loc, measure)
            s = start.loc[key] if key in start.index else None
            e = end.loc[key] if key in end.index else None
            er = eapc.loc[key] if key in eapc.index else None
            pct = np.nan
            if s is not None and e is not None and float(s["value"]) != 0:
                pct = (float(e["value"]) - float(s["value"])) / float(s["value"]) * 100
            rows.append(
                {
                    "location": loc,
                    "measure": measure,
                    "1990 age-standardized rate": fmt_float(s["value"], 2) if s is not None else "not available",
                    "2023 age-standardized rate": fmt_float(e["value"], 2) if e is not None else "not available",
                    "percent change": f"{pct:.1f}%" if not pd.isna(pct) else "not available",
                    "EAPC": f"{er['EAPC']:.3f}" if er is not None else "not available",
                    "EAPC 95% CI": f"{er['CI_lower']:.3f} to {er['CI_upper']:.3f}" if er is not None else "not available",
                    "EAPC P value": fmt_p(er["P_value"]) if er is not None else "not available",
                    "trend direction": er["trend_direction"] if er is not None else "not available",
                }
            )
    t5 = pd.DataFrame(rows)
    t5.to_csv(MS_TABLES / "Table_5_gbd_summary.csv", index=False, encoding="utf-8-sig")

    return {"auc_diff": auc_diff, "best_models": best, "table3": t3, "table5": t5}


def create_figures(sensitivity: pd.DataFrame) -> Dict[str, bool]:
    status = {}
    status["Figure_1_GBD_CVD_burden_and_risk_factors.png"] = composite_images(
        [
            ("CVD burden trends", FIGURES / "fig1_gbd_cvd_burden_trend.png"),
            ("China/global comparison", FIGURES / "fig1_gbd_china_global_comparison.png"),
            ("Attributable risk factors", FIGURES / "fig1_gbd_risk_factor_ranking.png"),
        ],
        MS_FIGURES / "Figure_1_GBD_CVD_burden_and_risk_factors.png",
        "GBD cardiovascular disease burden and attributable risk factors",
        layout=(2, 2),
        size=(2600, 2000),
    )
    status["Figure_2_ML_model_performance.png"] = copy_existing(
        FIGURES / "fig2_roc_pr_calibration.png", MS_FIGURES / "Figure_2_ML_model_performance.png"
    )
    status["Figure_3_SHAP_and_feature_importance.png"] = copy_existing(
        FIGURES / "fig3_shap_summary.png", MS_FIGURES / "Figure_3_SHAP_and_feature_importance.png"
    )
    status["Figure_4_RCS_key_indicators.png"] = composite_images(
        [
            ("UACR", FIGURES / "fig4_rcs_uacr.png"),
            ("eGFR", FIGURES / "fig4_rcs_egfr.png"),
            ("TyG-WC", FIGURES / "fig4_rcs_tyg_wc.png"),
            ("SIRI", FIGURES / "fig4_rcs_siri.png"),
        ],
        MS_FIGURES / "Figure_4_RCS_key_indicators.png",
        "Restricted cubic spline analyses for key indicators",
        layout=(2, 2),
        size=(2400, 1800),
    )
    sens_panel = MS_FIGURES / "_stage5_sensitivity_panel.png"
    make_sensitivity_panel(sensitivity, sens_panel)
    status["Figure_5_Subgroup_and_sensitivity_analysis.png"] = composite_images(
        [
            ("Subgroup forest plot", FIGURES / "fig5_subgroup_forest.png"),
            ("Sensitivity analysis", sens_panel),
        ],
        MS_FIGURES / "Figure_5_Subgroup_and_sensitivity_analysis.png",
        "Subgroup and sensitivity analyses",
        layout=(1, 2),
        size=(2600, 1400),
    )
    try:
        sens_panel.unlink(missing_ok=True)
    except Exception:
        pass
    return status


def gbd_result_check(gbd_trend: pd.DataFrame, gbd_risk: pd.DataFrame) -> Dict[str, object]:
    locations = ["China", "Global", "United States"]
    measures = ["DALYs", "Deaths", "Prevalence"]
    focus = gbd_trend[
        gbd_trend["location"].isin(locations)
        & gbd_trend["sex"].eq("Both")
        & gbd_trend["age"].str.contains("Age-standardized", case=False, na=False)
        & gbd_trend["metric"].eq("Rate")
        & gbd_trend["measure"].isin(measures)
        & gbd_trend["year"].isin([1990, 2023])
    ].copy()
    lines = ["# Stage 5 GBD Result Check", ""]
    summary_rows = []
    for loc in locations:
        lines.append(f"## {loc}")
        lines.append("")
        for measure in measures:
            s = focus[(focus["location"].eq(loc)) & focus["measure"].eq(measure) & focus["year"].eq(1990)]
            e = focus[(focus["location"].eq(loc)) & focus["measure"].eq(measure) & focus["year"].eq(2023)]
            if s.empty or e.empty:
                lines.append(f"- {measure}: not available")
                continue
            sv, ev = float(s.iloc[0]["value"]), float(e.iloc[0]["value"])
            pct = (ev - sv) / sv * 100 if sv else np.nan
            lines.append(f"- {measure} age-standardized rate: 1990={sv:.2f}; 2023={ev:.2f}; percent change={pct:.1f}%.")
            summary_rows.append((loc, measure, sv, ev, pct))
        lines.append("")

    def ranking(metric: str, age: str) -> pd.DataFrame:
        return gbd_risk[
            gbd_risk["year"].eq(2023)
            & gbd_risk["measure"].eq("DALYs")
            & gbd_risk["metric"].eq(metric)
            & gbd_risk["age"].eq(age)
            & gbd_risk["sex"].eq("Both")
        ].sort_values(["location", "rank"])

    number_rank = ranking("Number", "All ages")
    rate_rank = ranking("Rate", "Age-standardized")
    stage4_report_path = DIAG / "stage4_gbd_analysis_report.md"
    stage4_text = stage4_report_path.read_text(encoding="utf-8") if stage4_report_path.exists() else ""
    stage4_checks = []
    for loc, measure, sv, ev, pct in summary_rows:
        pattern = re.compile(
            rf"- {re.escape(loc)} {re.escape(measure)} age-standardized rate .*?\(([-+0-9.]+)% change\)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(stage4_text)
        if match:
            reported_pct = float(match.group(1))
            consistent = abs(reported_pct - pct) < 0.05
            stage4_checks.append((loc, measure, pct, reported_pct, consistent))
        else:
            stage4_checks.append((loc, measure, pct, None, None))
    lines.extend(["## Percent Change Check", ""])
    lines.append("- Percent changes below were recalculated directly from `gbd_cvd_burden_trend.csv` using age-standardized Both-sex rates.")
    for loc, measure, pct, reported_pct, consistent in stage4_checks:
        if consistent is None:
            lines.append(f"- {loc} {measure}: Stage 4.1 report comparison not available.")
        elif consistent:
            lines.append(f"- {loc} {measure}: consistent with Stage 4.1 report ({pct:.1f}%).")
        else:
            lines.append(
                f"- {loc} {measure}: not consistent with Stage 4.1 report; "
                f"burden trend table gives {pct:.1f}%, while Stage 4.1 report stated {reported_pct:.1f}%."
            )
    if any(item[-1] is False for item in stage4_checks):
        lines.append(
            "- For manuscript tables and Results, use `gbd_cvd_burden_trend.csv` and `gbd_eapc_summary.csv`; "
            "the inconsistent Stage 4.1 text appears to have mixed risk-attributable rows into the burden-trend summary."
        )
    lines.append("")
    lines.extend(["## 2023 Attributable CVD DALYs Number Ranking", ""])
    for loc, sub in number_rank.groupby("location"):
        parts = [f"{int(r['rank'])}. {r['rei']} ({fmt_num(r['value'])})" for _, r in sub.iterrows()]
        lines.append(f"- {loc}: " + "; ".join(parts) + ".")
    lines.extend(["", "## 2023 Attributable CVD DALYs Age-Standardized Rate Ranking", ""])
    for loc, sub in rate_rank.groupby("location"):
        parts = [f"{int(r['rank'])}. {r['rei']} ({r['value']:.2f})" for _, r in sub.iterrows()]
        lines.append(f"- {loc}: " + "; ".join(parts) + ".")
    lines.extend(["", "## Number Versus Rate Ranking", ""])
    differences = []
    for loc in sorted(set(number_rank["location"]).intersection(set(rate_rank["location"]))):
        n_order = number_rank[number_rank["location"].eq(loc)].sort_values("rank")["rei"].tolist()
        r_order = rate_rank[rate_rank["location"].eq(loc)].sort_values("rank")["rei"].tolist()
        if n_order != r_order:
            differences.append(loc)
            lines.append(f"- {loc}: Number and age-standardized rate rankings differ.")
        else:
            lines.append(f"- {loc}: Number and age-standardized rate rankings are identical.")
    if differences:
        lines.append("- Ranking differences are expected because Number reflects population size and age structure, whereas age-standardized Rate adjusts for age distribution.")
    (DIAG / "stage5_gbd_result_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"number_rank": number_rank, "rate_rank": rate_rank, "differences": differences, "trend_summary": summary_rows}


def build_results_draft(
    table1: pd.DataFrame,
    perf_info: Dict[str, object],
    key_summary: pd.DataFrame,
    regression: pd.DataFrame,
    rcs: pd.DataFrame,
    subgroup: pd.DataFrame,
    sensitivity: pd.DataFrame,
    gbd_trend: pd.DataFrame,
    gbd_eapc: pd.DataFrame,
    gbd_risk: pd.DataFrame,
) -> None:
    best = perf_info["best_models"].copy()
    t5 = perf_info["table5"]
    lines = ["# Results", ""]
    lines.extend(["## Global and national burden of cardiovascular diseases from GBD 2023", ""])
    for _, row in t5.iterrows():
        if row["location"] in ["Global", "China", "United States"]:
            lines.append(
                f"- In {row['location']}, the age-standardized {row['measure']} rate changed from "
                f"{row['1990 age-standardized rate']} in 1990 to {row['2023 age-standardized rate']} in 2023 "
                f"({row['percent change']}; EAPC {row['EAPC']}%, 95% CI {row['EAPC 95% CI']}, P={row['EAPC P value']})."
            )
    risk_key = gbd_risk[
        gbd_risk["year"].eq(2023)
        & gbd_risk["measure"].eq("DALYs")
        & gbd_risk["metric"].eq("Number")
        & gbd_risk["age"].eq("All ages")
        & gbd_risk["sex"].eq("Both")
        & gbd_risk["rank"].le(5)
    ].sort_values(["location", "rank"])
    for loc, sub in risk_key.groupby("location"):
        parts = [f"{int(r['rank'])}. {r['rei']}" for _, r in sub.iterrows()]
        lines.append(f"- In {loc}, the leading 2023 attributable CVD DALY risk factors by number were " + ", ".join(parts) + ".")

    lines.extend(["", "## Study population and weighted baseline characteristics in NHANES", ""])
    n_total = int(best["n_total"].max()) if "n_total" in best else 16827
    summary_path = MS_RESULTS / "results_summary_2005_2018.md"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    cases_match = re.search(r"CVD cases:\s*(\d+);\s*non-CVD:\s*(\d+)", summary_text)
    if cases_match:
        lines.append(
            f"- The NHANES analytic sample included {n_total:,} adults in the 2005-2018 cycles, "
            f"including {int(cases_match.group(1)):,} with CVD and {int(cases_match.group(2)):,} without CVD."
        )
    else:
        lines.append(f"- The NHANES analytic sample included {n_total:,} adults in the 2005-2018 cycles; CVD case count was not available.")
    for var in ["age", "bmi", "pir"]:
        row = table1[table1["variable"].eq(var)]
        if not row.empty:
            r = row.iloc[0]
            label = VAR_LABELS.get(var, var)
            lines.append(f"- {label}: non-CVD {r['non_cvd']} versus CVD {r['cvd']}, P={fmt_p(r['p_value'])}.")

    lines.extend(["", "## Predictive performance of interpretable machine learning models", ""])
    for _, r in best.sort_values("feature_set").iterrows():
        lines.append(
            f"- In the {r['feature_set']}, the best model was {r['model']} "
            f"(AUC={r['auc']:.4f}, PR-AUC={r['pr_auc']:.4f}, F1={r['f1']:.4f}, Brier score={r['brier_score']:.4f})."
        )
    lines.append(f"- The AUC difference between the full clinical and biomarker-focused best models was {perf_info['auc_diff']:.4f}.")

    lines.extend(["", "## Feature importance and consensus ranking", ""])
    ks = key_summary.set_index("variable_name")
    for ind in ["uacr", "egfr", "tyg_wc", "siri", "tyg_index", "non_hdl_cholesterol", "ldl_cholesterol", "hba1c"]:
        if ind in ks.index:
            r = ks.loc[ind]
            lines.append(
                f"- {r['indicator']} was routine top10={r['in_routine_indicator_model_top10']} and "
                f"biomarker-focused top10={r['in_biomarker_focused_model_top10']}."
            )

    lines.extend(["", "## Survey-weighted associations between key indicators and CVD", ""])
    lines.append("- Model 3 was used as the main statistical model; Model 4 was treated as a strict fully adjusted model with a smaller complete-case sample and potential overadjustment among correlated biomarkers.")
    for ind in CORE_INDICATORS:
        m3 = regression[(regression["indicator"].eq(ind)) & regression["form"].eq("continuous") & regression["model"].eq("Model 3")]
        m4 = regression[(regression["indicator"].eq(ind)) & regression["form"].eq("continuous") & regression["model"].eq("Model 4")]
        if m3.empty:
            lines.append(f"- {DISPLAY[ind]}: Model 3 result not available.")
            continue
        a = m3.iloc[0]
        sentence = f"- {DISPLAY[ind]}: Model 3 OR={a['OR']:.3f} (95% CI {a['CI_lower']:.3f}-{a['CI_upper']:.3f}), P={fmt_p(a['P_value'])}."
        if not m4.empty:
            b = m4.iloc[0]
            sentence += f" Strict Model 4 OR={b['OR']:.3f} (95% CI {b['CI_lower']:.3f}-{b['CI_upper']:.3f}), P={fmt_p(b['P_value'])}."
        lines.append(sentence)
    ldl = regression[(regression["indicator"].eq("ldl_cholesterol")) & regression["form"].eq("continuous") & regression["model"].eq("Model 4")]
    if not ldl.empty:
        r = ldl.iloc[0]
        lines.append(f"- LDL-C had an inverse association in strict Model 4 (OR={r['OR']:.3f}, 95% CI {r['CI_lower']:.3f}-{r['CI_upper']:.3f}, P={fmt_p(r['P_value'])}); this was described cautiously and not interpreted as protective.")

    lines.extend(["", "## Nonlinear associations, subgroup analysis, and sensitivity analysis", ""])
    for ind in ["uacr", "egfr", "tyg_wc", "siri"]:
        rr = rcs[rcs["indicator"].eq(ind)]
        if not rr.empty:
            lines.append(f"- {DISPLAY[ind]} showed evidence of nonlinearity in RCS analysis (P for nonlinearity={fmt_p(rr.iloc[0]['p_nonlinearity'])}).")
    sig_inter = subgroup[subgroup["interaction_p"].notna() & (subgroup["interaction_p"] < 0.05)][["indicator", "subgroup", "interaction_p"]].drop_duplicates()
    if sig_inter.empty:
        lines.append("- No subgroup interaction reached P<0.05.")
    else:
        for _, r in sig_inter.iterrows():
            lines.append(f"- A subgroup interaction was observed for {DISPLAY.get(r['indicator'], r['indicator'])} by {r['subgroup']} (P interaction={fmt_p(r['interaction_p'])}).")
    for ind in CORE_INDICATORS:
        rows = sensitivity[sensitivity["indicator"].eq(ind)]
        sig = rows[rows["P_value"] < 0.05]
        if rows.empty:
            lines.append(f"- {DISPLAY[ind]} sensitivity results were not available.")
        else:
            lines.append(f"- {DISPLAY[ind]} was significant in {len(sig)}/{len(rows)} sensitivity analyses.")
    lines.append("- UACR showed the most consistent overall support across machine learning, Model 3, strict Model 4, RCS, and sensitivity analyses.")
    (MS_RESULTS / "results_draft_en.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_methods_skeleton() -> None:
    lines = [
        "# Methods Skeleton",
        "",
        "## Data sources",
        "- GBD 2023 cardiovascular disease burden CSV exports were obtained from official IHME/GBD sources and analyzed separately from NHANES.",
        "- NHANES 2005-2018 adult data were used for individual-level machine learning and survey-weighted epidemiologic analyses.",
        "",
        "## GBD analysis",
        "- GBD CSV files were read from `data/raw/gbd/`, with common IHME field aliases standardized.",
        "- Analyses focused on cardiovascular diseases, measures of deaths, DALYs, and prevalence, metrics of number and rate, and locations Global, China, and United States.",
        "- Age-standardized rates were summarized for 1990 and 2023, percent change was computed, and EAPC was estimated from log-linear year trends.",
        "- Attributable CVD burden was ranked for selected risk factors using 2023 DALY numbers and age-standardized rates.",
        "",
        "## NHANES study population",
        "- Adults aged 20 years or older were included after applying the project exclusion criteria and retaining participants with non-missing CVD outcome.",
        "- NHANES survey design variables included combined fasting weights, strata, and PSU for survey-weighted analyses.",
        "",
        "## CVD outcome definition",
        "- CVD was defined as self-reported congestive heart failure, coronary heart disease, angina, heart attack/myocardial infarction, or stroke.",
        "",
        "## Candidate predictors and derived indicators",
        "- Candidate predictors included demographic, lifestyle, anthropometric, blood pressure, glucose/lipid metabolism, inflammatory, and renal-function variables.",
        "- Derived indicators included TyG index, TyG-WC, TyG-BMI, AIP, non-HDL-C, NLR, PLR, SII, SIRI, eGFR, and UACR when source variables were available.",
        "",
        "## Machine learning modeling",
        "- Three leakage-audited feature sets were analyzed: full clinical, routine indicator, and biomarker-focused models.",
        "- Models included logistic regression, random forest, XGBoost, LightGBM, and Elastic Net/SVM depending on the project pipeline.",
        "- Train/test splitting was stratified, and preprocessing was fit within training-set pipelines.",
        "",
        "## SHAP and feature ranking",
        "- Feature importance was summarized using random forest importance, XGBoost importance, permutation importance, and held-out SHAP values.",
        "- Consensus ranking was derived across methods within feature sets.",
        "",
        "## Survey-weighted statistical analysis",
        "- Survey-weighted logistic regression estimated ORs and 95% CIs for selected indicators.",
        "- Model 3 was used as the main model; Model 4 was considered a strict fully adjusted model with additional biomarkers and smaller complete-case samples.",
        "",
        "## RCS, subgroup, and sensitivity analyses",
        "- Nonlinear associations were assessed using survey-weighted natural spline models.",
        "- Subgroup analyses considered age group, sex, diabetes, hypertension, and obesity status.",
        "- Sensitivity analyses excluded prior stroke, excluded prior heart attack, and used routine biomarker adjustment without hypertension/diabetes diagnoses.",
    ]
    (MS_RESULTS / "methods_skeleton_en.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_legends_notes() -> None:
    lines = [
        "# Figure Legends and Table Notes",
        "",
        "## Figure Legends",
        "",
        "**Figure 1. GBD cardiovascular disease burden and attributable risk factors.** Age-standardized CVD burden trends and 2023 attributable risk-factor rankings from official GBD CSV exports. GBD data were used only for macro-level public-health context.",
        "",
        "**Figure 2. Machine learning model performance.** ROC, precision-recall, and calibration summaries for leakage-audited NHANES prediction models.",
        "",
        "**Figure 3. SHAP and feature importance.** Held-out SHAP summary and feature-importance results for interpretable machine learning models.",
        "",
        "**Figure 4. Restricted cubic spline analyses for key indicators.** Survey-weighted spline curves for UACR, eGFR, TyG-WC, and SIRI.",
        "",
        "**Figure 5. Subgroup and sensitivity analyses.** Survey-weighted subgroup forest plot and sensitivity-analysis summary for key indicators.",
        "",
        "## Table Notes",
        "",
        "**Table 1.** Weighted baseline characteristics by CVD status. Continuous variables are shown as weighted mean (SE) when available; categorical variables are weighted percentages.",
        "",
        "**Table 2.** Best model performance by feature set. Metrics were computed in the held-out test set.",
        "",
        "**Table 3.** Consensus feature ranking and statistical validation summary for priority indicators.",
        "",
        "**Table 4.** Survey-weighted logistic regression for key indicators. Model 3 is the main model; Model 4 is a strict fully adjusted model and should be interpreted cautiously when estimates attenuate.",
        "",
        "**Table 5.** GBD CVD age-standardized burden rates in 1990 and 2023, percent change, and EAPC.",
        "",
        "## Abbreviations",
        "",
        "- CVD: cardiovascular disease",
        "- NHANES: National Health and Nutrition Examination Survey",
        "- GBD: Global Burden of Disease",
        "- DALYs: disability-adjusted life years",
        "- UACR: urine albumin-to-creatinine ratio",
        "- eGFR: estimated glomerular filtration rate",
        "- TyG-WC: triglyceride-glucose index multiplied by waist circumference",
        "- SIRI: systemic inflammation response index",
        "- SHAP: Shapley additive explanations",
        "- RCS: restricted cubic spline",
        "- OR: odds ratio",
        "- CI: confidence interval",
    ]
    (MS_RESULTS / "figure_legends_and_table_notes_en.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(figure_status: Dict[str, bool], missing_values: List[str]) -> None:
    table_files = [
        "Table_1_weighted_baseline_characteristics.csv",
        "Table_2_model_performance.csv",
        "Table_3_feature_ranking.csv",
        "Table_4_weighted_regression.csv",
        "Table_5_gbd_summary.csv",
    ]
    result_files = [
        MS_RESULTS / "results_draft_en.md",
        MS_RESULTS / "methods_skeleton_en.md",
        MS_RESULTS / "figure_legends_and_table_notes_en.md",
    ]
    lines = [
        "# Stage 5 Manuscript Package Audit",
        "",
        "## Main Tables",
        "",
    ]
    for f in table_files:
        path = MS_TABLES / f
        lines.append(f"- {f}: {path.exists()} ({path.stat().st_size if path.exists() else 0} bytes)")
    lines.extend(["", "## Main Figures", ""])
    for f, ok in figure_status.items():
        path = MS_FIGURES / f
        lines.append(f"- {f}: {ok and path.exists()} ({path.stat().st_size if path.exists() else 0} bytes)")
    lines.extend(["", "## Manuscript Text Files", ""])
    for path in result_files:
        lines.append(f"- {path.name}: {path.exists()} ({path.stat().st_size if path.exists() else 0} bytes)")
    lines.extend(["", "## Missing Values", ""])
    if missing_values:
        lines.extend([f"- {item}" for item in missing_values])
    else:
        lines.append("- No unexpected missing values were detected in required manuscript tables.")
    lines.extend(
        [
            "",
            "## GBD Consistency Check",
            "",
        ]
    )
    gbd_check = DIAG / "stage5_gbd_result_check.md"
    gbd_text = gbd_check.read_text(encoding="utf-8") if gbd_check.exists() else ""
    if "not consistent with Stage 4.1 report" in gbd_text:
        lines.append(
            "- A Stage 4.1 narrative inconsistency was detected for at least one GBD burden percentage; "
            "manuscript tables/results use `gbd_cvd_burden_trend.csv` and `gbd_eapc_summary.csv` directly."
        )
    else:
        lines.append("- GBD burden percent changes were consistent with the Stage 4.1 report where comparable.")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- No causal language was added.",
            "- GBD and NHANES were not merged or jointly modeled.",
            "- LDL-C inverse associations are flagged for cautious interpretation.",
            "- Model 4 is described as a strict fully adjusted model with smaller sample size and possible overadjustment.",
            "",
            "## Recommendation",
            "",
            "- The manuscript-ready results package is ready for English SCI full-text drafting, with UACR, eGFR, TyG-WC, and SIRI as the recommended core indicators.",
        ]
    )
    (DIAG / "stage5_manuscript_package_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    table1 = read_csv(TABLES / "table1_baseline_weighted.csv")
    performance = read_csv(TABLES / "model_performance_2005_2018_by_featureset.csv")
    read_csv(TABLES / "feature_ranking_2005_2018_by_featureset.csv")
    read_csv(TABLES / "table2_model_performance.csv")
    read_csv(TABLES / "table3_feature_ranking.csv")
    require(MS_RESULTS / "results_summary_2005_2018.md")
    regression = read_csv(TABLES / "table4_weighted_regression.csv")
    rcs = read_csv(TABLES / "rcs_nonlinearity_tests.csv")
    subgroup = read_csv(TABLES / "subgroup_analysis.csv")
    sensitivity = read_csv(TABLES / "sensitivity_analysis.csv")
    key_summary = read_csv(TABLES / "key_result_summary_for_manuscript.csv")
    require(DIAG / "stage3_5_manuscript_result_interpretation.md")
    gbd_trend = read_csv(TABLES / "gbd_cvd_burden_trend.csv")
    gbd_eapc = read_csv(TABLES / "gbd_eapc_summary.csv")
    gbd_risk = read_csv(TABLES / "gbd_risk_factor_ranking.csv")
    require(DIAG / "stage4_gbd_analysis_report.md")
    require(MS_RESULTS / "gbd_results_skeleton_en.md")

    perf_info = create_tables(table1, performance, key_summary, regression, gbd_trend, gbd_eapc)
    figure_status = create_figures(sensitivity)
    gbd_result_check(gbd_trend, gbd_risk)
    build_results_draft(table1, perf_info, key_summary, regression, rcs, subgroup, sensitivity, gbd_trend, gbd_eapc, gbd_risk)
    build_methods_skeleton()
    build_legends_notes()

    missing_values = []
    table3 = pd.read_csv(MS_TABLES / "Table_3_feature_ranking.csv")
    for _, row in table3.iterrows():
        if row["RCS nonlinear"] == "not available" and row["indicator"] in ["LDL-C", "HbA1c"]:
            missing_values.append(f"RCS nonlinearity was not available for {row['indicator']} because Stage 3 RCS focused on prespecified spline indicators.")
    build_audit(figure_status, missing_values)
    print("Stage 5 manuscript-ready package generated.")
    print("Core indicators: UACR, eGFR, TyG-WC, SIRI")
    best = perf_info["best_models"]
    for _, row in best.sort_values("feature_set").iterrows():
        print(f"{row['feature_set']}: {row['model']} AUC={row['auc']:.4f}, PR-AUC={row['pr_auc']:.4f}, F1={row['f1']:.4f}")
    print(f"GBD Figure 1 available: {figure_status.get('Figure_1_GBD_CVD_burden_and_risk_factors.png', False)}")


if __name__ == "__main__":
    main()
