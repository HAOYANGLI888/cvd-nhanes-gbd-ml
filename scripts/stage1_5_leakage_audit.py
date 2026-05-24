import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.feature_sets import (  # noqa: E402
    CVD_OUTCOME_COMPONENTS,
    CVD_DERIVED_OUTCOME_FIELDS,
    POTENTIAL_LEAKAGE_FIELDS,
    build_predictor_feature_map,
    feature_sets_for_columns,
    sanitized_features,
)
from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402
from cvd_nhanes_gbd.model_utils import (  # noqa: E402
    build_model_configs,
    evaluate_binary_classifier,
    get_preprocessed_feature_names,
    make_model_pipeline,
    make_preprocessor,
    optional_package_available,
    predict_probability,
)
from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1.5 leakage audit and feature-set modeling.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input", type=Path, default=None, help="Default: data/processed/nhanes_filtered_pre_imputation.csv")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--shap-sample", type=int, default=1000)
    parser.add_argument("--models", default="Logistic Regression,Random Forest,XGBoost,LightGBM,SVM,Elastic Net")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def canonical_feature_name(name: str, original_features: Iterable[str]) -> str:
    for feature in sorted(original_features, key=len, reverse=True):
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def aggregate_to_original(importances: pd.Series, original_features: List[str]) -> pd.Series:
    rows = []
    for feature_name, value in importances.items():
        rows.append({"feature": canonical_feature_name(str(feature_name), original_features), "importance": abs(float(value))})
    frame = pd.DataFrame(rows)
    return frame.groupby("feature", as_index=True)["importance"].sum().sort_values(ascending=False)


def rank_series(importances: pd.Series, method: str, feature_set: str) -> pd.DataFrame:
    imp = importances.replace([np.inf, -np.inf], np.nan).dropna()
    imp = imp[imp > 0]
    if imp.empty:
        return pd.DataFrame(columns=["feature_set", "feature", "method", "importance", "rank"])
    ranked = imp.sort_values(ascending=False).reset_index()
    ranked.columns = ["feature", "importance"]
    ranked.insert(0, "feature_set", feature_set)
    ranked["method"] = method
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked[["feature_set", "feature", "method", "importance", "rank"]]


def build_consensus(long_table: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if long_table.empty:
        return pd.DataFrame()
    rows = []
    for feature_set, group in long_table.groupby("feature_set"):
        top_hits = group.loc[group["rank"] <= top_n]
        summary = (
            group.groupby("feature")
            .agg(
                mean_rank=("rank", "mean"),
                median_rank=("rank", "median"),
                methods_present=("method", lambda x: ";".join(sorted(set(x)))),
            )
            .reset_index()
        )
        hit_counts = top_hits.groupby("feature")["method"].nunique().rename("top10_method_count")
        summary = summary.merge(hit_counts, on="feature", how="left")
        summary["top10_method_count"] = summary["top10_method_count"].fillna(0).astype(int)
        summary = summary.sort_values(["top10_method_count", "mean_rank"], ascending=[False, True]).reset_index(drop=True)
        summary.insert(0, "consensus_rank", np.arange(1, len(summary) + 1))
        summary.insert(0, "feature_set", feature_set)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def transformed_feature_importance(
    fitted_pipeline: Pipeline,
    original_features: List[str],
    importances: np.ndarray,
) -> pd.Series:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    feature_names = get_preprocessed_feature_names(preprocessor)
    return aggregate_to_original(pd.Series(importances, index=feature_names), original_features)


def explain_feature_set(
    feature_set_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int,
    shap_sample: int,
    figures_dir: Path,
    logger,
) -> pd.DataFrame:
    original_features = list(X_train.columns)
    rf = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    rf_pipe = Pipeline([("preprocessor", make_preprocessor(X_train, scale_numeric=False)), ("model", rf)])
    rf_pipe.fit(X_train, y_train)

    method_tables = []
    method_tables.append(
        rank_series(
            transformed_feature_importance(rf_pipe, original_features, rf_pipe.named_steps["model"].feature_importances_),
            "Random Forest train importance",
            feature_set_name,
        )
    )

    perm = permutation_importance(
        rf_pipe,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
    )
    method_tables.append(rank_series(pd.Series(perm.importances_mean, index=original_features), "Permutation held-out test", feature_set_name))

    if optional_package_available("xgboost"):
        try:
            from xgboost import XGBClassifier

            xgb_pipe = Pipeline(
                [
                    ("preprocessor", make_preprocessor(X_train, scale_numeric=False)),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=400,
                            learning_rate=0.03,
                            max_depth=3,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            xgb_pipe.fit(X_train, y_train)
            method_tables.append(
                rank_series(
                    transformed_feature_importance(xgb_pipe, original_features, xgb_pipe.named_steps["model"].feature_importances_),
                    "XGBoost train importance",
                    feature_set_name,
                )
            )
        except Exception as exc:
            logger.warning("XGBoost importance failed for %s: %s", feature_set_name, exc)

    if optional_package_available("shap"):
        try:
            import shap

            preprocessor = rf_pipe.named_steps["preprocessor"]
            model = rf_pipe.named_steps["model"]
            X_test_array = preprocessor.transform(X_test)
            feature_names = get_preprocessed_feature_names(preprocessor)
            rng = np.random.default_rng(random_state)
            if X_test_array.shape[0] > shap_sample:
                sample_idx = rng.choice(X_test_array.shape[0], size=shap_sample, replace=False)
            else:
                sample_idx = np.arange(X_test_array.shape[0])
            X_sample = np.asarray(X_test_array)[sample_idx]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif getattr(shap_values, "ndim", 0) == 3:
                shap_matrix = shap_values[:, :, 1]
            else:
                shap_matrix = shap_values
            mean_abs = pd.Series(np.abs(shap_matrix).mean(axis=0), index=feature_names)
            shap_rank = aggregate_to_original(mean_abs, original_features)
            method_tables.append(rank_series(shap_rank, "SHAP held-out test", feature_set_name))

            if feature_set_name == "biomarker_focused_model":
                shap.summary_plot(shap_matrix, X_sample, feature_names=feature_names, show=False, max_display=20)
                plt.tight_layout()
                plt.savefig(figures_dir / "fig3_shap_summary.png", dpi=300, bbox_inches="tight")
                plt.savefig(figures_dir / "stage1_5_biomarker_focused_shap_summary.png", dpi=300, bbox_inches="tight")
                plt.close()
        except Exception as exc:
            logger.warning("SHAP failed for %s: %s", feature_set_name, exc)

    return pd.concat([table for table in method_tables if not table.empty], ignore_index=True)


def train_models_by_feature_set(
    feature_set_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    configs = build_model_configs(y_train, args.random_state, logger)
    configs = {name: model for name, model in configs.items() if name in requested}
    rows = []
    prediction_rows = []
    for model_name, estimator in configs.items():
        pipeline = make_model_pipeline(model_name, estimator, X_train)
        try:
            pipeline.fit(X_train, y_train)
            y_prob = predict_probability(pipeline, X_test)
            metrics = evaluate_binary_classifier(y_test, y_prob)
        except Exception as exc:
            logger.warning("%s failed for %s: %s", model_name, feature_set_name, exc)
            continue
        row = {
            "feature_set": feature_set_name,
            "model": model_name,
            "n_total": len(y),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": X.shape[1],
            "stratified_split": True,
            "preprocessing_fit_scope": "training set only via sklearn Pipeline",
        }
        row.update(metrics)
        rows.append(row)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "observed_cvd": y_test.to_numpy(),
                    "predicted_probability": y_prob,
                }
            )
        )
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return pd.DataFrame(rows), predictions, X_train, X_test, y_train, y_test


def add_public_metric_columns(performance: pd.DataFrame) -> pd.DataFrame:
    out = performance.copy()
    out["AUC"] = out["auc"]
    out["PR-AUC"] = out["pr_auc"]
    out["F1"] = out["f1"]
    out["Brier score"] = out["brier_score"]
    return out


def write_stage2_curve_figure(performance: pd.DataFrame, predictions: pd.DataFrame, figures_dir: Path) -> None:
    if performance.empty or predictions.empty:
        return
    best = performance.sort_values(["feature_set", "auc"], ascending=[True, False]).groupby("feature_set").head(1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for row in best.itertuples(index=False):
        pred = predictions.loc[
            predictions["feature_set"].eq(row.feature_set) & predictions["model"].eq(row.model)
        ].copy()
        if pred.empty:
            continue
        y_true = pred["observed_cvd"].astype(int).to_numpy()
        y_prob = pred["predicted_probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, lw=2, label=f"{row.feature_set} ({row.model}) AUC={row.auc:.3f}")
        axes[1].plot(recall, precision, lw=2, label=f"{row.feature_set} PR-AUC={row.pr_auc:.3f}")
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        axes[2].plot(prob_pred, prob_true, marker="o", lw=1.8, label=row.feature_set)
    axes[0].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    axes[2].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    axes[0].set_title("ROC")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_title("Precision-recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[2].set_title("Calibration")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Observed probability")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_roc_pr_calibration.png", dpi=300)
    plt.close(fig)


def write_report(
    root: Path,
    feature_sets: Dict,
    removed: List[str],
    performance: pd.DataFrame,
    consensus: pd.DataFrame,
    all_feature_rows: pd.DataFrame,
) -> None:
    report = root / "outputs/diagnostics/stage1_5_leakage_audit_report.md"
    lines = [
        "# Stage 1.5 Leakage Audit Report",
        "",
        "No manuscript text and no fabricated results were generated. This report audits the 2015-2018 trial run and updates the main-analysis modeling plan.",
        "",
        "## CVD Outcome Definition",
        "",
        "CVD=1 is defined only from NHANES MCQ-derived fields for congestive heart failure, coronary heart disease, angina, heart attack/myocardial infarction, or stroke.",
        "",
        f"Outcome source fields: `{', '.join(CVD_OUTCOME_COMPONENTS)}`.",
        f"Derived outcome fields excluded from predictors: `{', '.join(CVD_DERIVED_OUTCOME_FIELDS)}`.",
        "",
        "## Leakage Findings",
        "",
    ]
    if removed:
        lines.append("Potential leakage or raw survey-code fields were present in the dataset but excluded from all Stage 1.5 feature sets:")
        for feature in removed:
            lines.append(f"- `{feature}`")
    else:
        lines.append("No leakage variables were present in the candidate predictor list.")
    any_included_leak = all_feature_rows.loc[
        all_feature_rows["outcome_or_potential_leakage"].eq(True)
        & (
            all_feature_rows["in_full_clinical_model"].eq(True)
            | all_feature_rows["in_routine_indicator_model"].eq(True)
            | all_feature_rows["in_biomarker_focused_model"].eq(True)
        )
    ]
    lines.append("")
    lines.append(f"Leakage variables included in final feature sets: **{len(any_included_leak)}**.")
    lines.extend(
        [
            "",
            "## Train/Test Protocol",
            "",
            "- Train/test split is stratified by CVD outcome.",
            "- Imputer and scaler are inside sklearn Pipelines and are fit on the training set only.",
            "- SMOTE is not used in this Stage 1.5 run; if enabled later, it is inside the training pipeline only.",
            "- Feature ranking is post-hoc model interpretation. No top-ranked feature subset is selected before splitting or used to refit the evaluated models.",
            "- SHAP uses a model trained on the training set and explains the held-out test set.",
            "",
            "## Feature Set Sample Sizes",
            "",
        ]
    )
    for name, feature_set in feature_sets.items():
        subset = performance.loc[performance["feature_set"].eq(name)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(f"- `{name}`: n_total={int(row['n_total'])}, n_train={int(row['n_train'])}, n_test={int(row['n_test'])}, n_features={int(row['n_features'])}")
    lines.extend(["", "## Model Performance", ""])
    best_rows = performance.sort_values(["feature_set", "auc"], ascending=[True, False]).groupby("feature_set").head(1)
    for row in best_rows.itertuples(index=False):
        lines.append(f"- `{row.feature_set}` best model: {row.model}; AUC={row.auc:.4f}, PR-AUC={row.pr_auc:.4f}, F1={row.f1:.4f}")

    lines.extend(["", "## Top 10 Features", ""])
    for feature_set, group in consensus.groupby("feature_set"):
        top = group.sort_values("consensus_rank").head(10)["feature"].tolist()
        lines.append(f"- `{feature_set}`: {', '.join(top)}")

    lines.extend(
        [
            "",
            "## Main-Analysis Recommendation",
            "",
            "Proceed to the 2005-2018 Python/NHANES main analysis with the corrected no-global-imputation ML path. R survey-weighted analyses still require Rscript to be installed or added to PATH.",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)
    logger = setup_logging("stage1_5_leakage_audit", root / "outputs/logs/stage1_5_leakage_audit.log", args.log_level)

    input_path = args.input or (root / "data/processed/nhanes_filtered_pre_imputation.csv")
    tables_dir = root / "outputs/tables"
    diagnostics_dir = root / "outputs/diagnostics"
    figures_dir = root / "outputs/figures"

    df = pd.read_csv(input_path)
    df = df.loc[df["cvd"].notna()].copy()
    y = pd.to_numeric(df["cvd"], errors="coerce").astype(int)
    if y.nunique() < 2:
        raise RuntimeError("CVD outcome has fewer than two classes.")

    feature_map = build_predictor_feature_map(df)
    feature_map.to_csv(diagnostics_dir / f"predictor_feature_list_{args.start_year}_{args.end_year}.csv", index=False, encoding="utf-8-sig")

    removed = sorted([feature for feature in df.columns if feature in POTENTIAL_LEAKAGE_FIELDS or feature.endswith("_code") or feature in {"seqn", "strata", "psu"}])
    feature_sets = feature_sets_for_columns(df.columns)

    performance_tables = []
    prediction_tables = []
    ranking_tables = []
    for feature_set_name in feature_sets:
        features = sanitized_features(df, feature_set_name)
        logger.info("Feature set %s: %s features", feature_set_name, len(features))
        if not features:
            logger.warning("Feature set %s has no features and will be skipped.", feature_set_name)
            continue
        X = df[features].copy()
        performance, predictions, X_train, X_test, y_train, y_test = train_models_by_feature_set(feature_set_name, X, y, args, logger)
        performance_tables.append(performance)
        prediction_tables.append(predictions)
        ranking_tables.append(
            explain_feature_set(feature_set_name, X_train, X_test, y_train, y_test, args.random_state, args.shap_sample, figures_dir, logger)
        )

    if not performance_tables:
        raise RuntimeError("No feature-set models completed.")
    performance_all = pd.concat(performance_tables, ignore_index=True)
    prediction_all = pd.concat([table for table in prediction_tables if not table.empty], ignore_index=True)
    ranking_long = pd.concat([table for table in ranking_tables if not table.empty], ignore_index=True)
    consensus = build_consensus(ranking_long, top_n=10)

    performance_public = add_public_metric_columns(performance_all)
    ranking_public = ranking_long.rename(columns={"method": "ranking_method", "importance": "importance_score"})
    performance_public.to_csv(tables_dir / f"model_performance_{args.start_year}_{args.end_year}_by_featureset.csv", index=False)
    ranking_public.to_csv(tables_dir / f"feature_ranking_{args.start_year}_{args.end_year}_by_featureset.csv", index=False)
    ranking_long.to_csv(tables_dir / f"feature_importance_{args.start_year}_{args.end_year}_by_featureset_long.csv", index=False)
    consensus.to_csv(tables_dir / f"feature_ranking_consensus_{args.start_year}_{args.end_year}_by_featureset.csv", index=False)
    performance_public.to_csv(tables_dir / "table2_model_performance.csv", index=False)
    consensus.to_csv(tables_dir / "table3_feature_ranking.csv", index=False)
    if not prediction_all.empty:
        prediction_all.to_csv(tables_dir / f"model_predictions_{args.start_year}_{args.end_year}_by_featureset.csv", index=False)
        write_stage2_curve_figure(performance_all, prediction_all, figures_dir)
    write_report(root, feature_sets, removed, performance_all, consensus, feature_map)
    logger.info("Stage 1.5 complete.")


if __name__ == "__main__":
    main()
