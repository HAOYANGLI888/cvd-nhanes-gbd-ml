import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402
from cvd_nhanes_gbd.model_utils import (  # noqa: E402
    class_balance_scale,
    get_preprocessed_feature_names,
    make_preprocessor,
    optional_package_available,
    split_features_target,
)
from cvd_nhanes_gbd.nhanes_metadata import TARGET_INDICATORS  # noqa: E402
from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consensus feature selection for NHANES CVD prediction.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input", type=Path, default=None, help="Default: data/processed/nhanes_filtered_pre_imputation.csv")
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-boruta-iter", type=int, default=100)
    parser.add_argument("--shap-sample", type=int, default=1000)
    parser.add_argument("--skip-boruta", action="store_true")
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


def rank_series(importances: pd.Series, method: str) -> pd.DataFrame:
    imp = importances.replace([np.inf, -np.inf], np.nan).dropna()
    imp = imp[imp > 0]
    if imp.empty:
        return pd.DataFrame(columns=["feature", "method", "importance", "rank"])
    ranked = imp.sort_values(ascending=False).reset_index()
    ranked.columns = ["feature", "importance"]
    ranked["method"] = method
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked[["feature", "method", "importance", "rank"]]


def lasso_importance(X_train: pd.DataFrame, y_train: pd.Series, original_features: List[str], random_state: int) -> pd.Series:
    preprocessor = make_preprocessor(X_train, scale_numeric=True)
    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        max_iter=8000,
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    feature_names = get_preprocessed_feature_names(pipe.named_steps["preprocessor"])
    coefs = pd.Series(np.abs(pipe.named_steps["model"].coef_[0]), index=feature_names)
    return aggregate_to_original(coefs, original_features)


def rf_importance(X_train: pd.DataFrame, y_train: pd.Series, original_features: List[str], random_state: int) -> pd.Series:
    preprocessor = make_preprocessor(X_train, scale_numeric=False)
    rf = RandomForestClassifier(
        n_estimators=700,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("model", rf)])
    pipe.fit(X_train, y_train)
    feature_names = get_preprocessed_feature_names(pipe.named_steps["preprocessor"])
    importances = pd.Series(pipe.named_steps["model"].feature_importances_, index=feature_names)
    return aggregate_to_original(importances, original_features)


def xgb_importance(X_train: pd.DataFrame, y_train: pd.Series, original_features: List[str], random_state: int, logger) -> pd.Series:
    if not optional_package_available("xgboost"):
        logger.warning("xgboost is unavailable; XGBoost importance skipped.")
        return pd.Series(dtype=float)
    from xgboost import XGBClassifier

    preprocessor = make_preprocessor(X_train, scale_numeric=False)
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=class_balance_scale(y_train),
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("model", xgb)])
    pipe.fit(X_train, y_train)
    feature_names = get_preprocessed_feature_names(pipe.named_steps["preprocessor"])
    importances = pd.Series(pipe.named_steps["model"].feature_importances_, index=feature_names)
    return aggregate_to_original(importances, original_features)


def boruta_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    original_features: List[str],
    random_state: int,
    max_iter: int,
    logger,
) -> pd.Series:
    if not optional_package_available("boruta"):
        logger.warning("boruta is unavailable; Boruta feature selection skipped.")
        return pd.Series(dtype=float)
    from boruta import BorutaPy

    preprocessor = make_preprocessor(X_train, scale_numeric=False)
    X_array = preprocessor.fit_transform(X_train)
    feature_names = get_preprocessed_feature_names(preprocessor)
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    selector = BorutaPy(rf, n_estimators="auto", max_iter=max_iter, random_state=random_state, verbose=0)
    selector.fit(np.asarray(X_array), y_train.to_numpy())
    scores = pd.Series(1 / selector.ranking_.astype(float), index=feature_names)
    scores.loc[selector.support_] = scores.max() + 1
    return aggregate_to_original(scores, original_features)


def permutation_importance_scores(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int,
) -> pd.Series:
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocessor", make_preprocessor(X_train, scale_numeric=False)), ("model", rf)])
    pipe.fit(X_train, y_train)
    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.Series(perm.importances_mean, index=X_train.columns).sort_values(ascending=False)


def shap_importance_and_plot(
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    y_train: pd.Series,
    y_explain: pd.Series,
    figures_dir: Path,
    original_features: List[str],
    random_state: int,
    sample_size: int,
    logger,
) -> pd.Series:
    if not optional_package_available("shap"):
        logger.warning("shap is unavailable; SHAP importance skipped.")
        return pd.Series(dtype=float)
    import shap

    preprocessor = make_preprocessor(X_train, scale_numeric=False)
    X_array = preprocessor.fit_transform(X_train)
    X_explain_array = preprocessor.transform(X_explain)
    feature_names = get_preprocessed_feature_names(preprocessor)
    rng = np.random.default_rng(random_state)
    if X_explain_array.shape[0] > sample_size:
        sample_idx = rng.choice(X_explain_array.shape[0], size=sample_size, replace=False)
    else:
        sample_idx = np.arange(X_explain_array.shape[0])

    # Use RandomForest for SHAP by default. It avoids version-sensitive XGBoost
    # model-dump parsing while still providing tree-based SHAP values.
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_array, y_train)
    X_sample = np.asarray(X_explain_array)[sample_idx]
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif getattr(shap_values, "ndim", 0) == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values
    except Exception as exc:
        logger.warning("Tree SHAP failed; using permutation importance as SHAP fallback: %s", exc)
        perm = permutation_importance(model, X_sample, y_explain.to_numpy()[sample_idx], scoring="roc_auc", n_repeats=5, random_state=random_state, n_jobs=-1)
        fallback = pd.Series(perm.importances_mean, index=feature_names)
        fallback_ranked = aggregate_to_original(fallback, original_features)
        plt.figure(figsize=(8, 6))
        fallback_ranked.head(20).sort_values().plot(kind="barh")
        plt.xlabel("Fallback permutation importance")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig3_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        return fallback_ranked

    mean_abs = pd.Series(np.abs(shap_matrix).mean(axis=0), index=feature_names)
    try:
        shap.summary_plot(shap_matrix, X_sample, feature_names=feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(figures_dir / "fig3_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.warning("Could not draw SHAP summary plot: %s", exc)
        fallback = aggregate_to_original(mean_abs, original_features).head(20).sort_values()
        plt.figure(figsize=(8, 6))
        fallback.plot(kind="barh")
        plt.xlabel("Mean absolute SHAP value")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig3_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
    return aggregate_to_original(mean_abs, original_features)


def build_consensus(long_table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if long_table.empty:
        return pd.DataFrame()
    top_hits = long_table.loc[long_table["rank"] <= top_n].copy()
    pivot_rank = long_table.pivot_table(index="feature", columns="method", values="rank", aggfunc="min")
    summary = (
        long_table.groupby("feature")
        .agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), methods_present=("method", lambda x: ";".join(sorted(set(x)))))
        .reset_index()
    )
    hit_counts = top_hits.groupby("feature")["method"].nunique().rename("top10_method_count")
    summary = summary.merge(hit_counts, on="feature", how="left")
    summary["top10_method_count"] = summary["top10_method_count"].fillna(0).astype(int)
    for method in pivot_rank.columns:
        summary[f"{method.lower().replace(' ', '_')}_rank"] = summary["feature"].map(pivot_rank[method])
    target_set = set(TARGET_INDICATORS)
    summary["priority_metabolic_inflammatory_renal_indicator"] = summary["feature"].isin(target_set)
    summary = summary.sort_values(
        ["top10_method_count", "mean_rank", "priority_metabolic_inflammatory_renal_indicator"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    summary.insert(0, "consensus_rank", np.arange(1, len(summary) + 1))
    return summary


def plot_consensus_bar(consensus: pd.DataFrame, figures_dir: Path) -> None:
    if consensus.empty:
        return
    top = consensus.head(20).sort_values("top10_method_count")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top, x="top10_method_count", y="feature", hue="priority_metabolic_inflammatory_renal_indicator", dodge=False)
    plt.xlabel("Number of methods ranking feature in top 10")
    plt.ylabel("")
    plt.legend(title="Priority indicator", loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_consensus_top20.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)
    logger = setup_logging("feature_selection", root / "outputs/logs/feature_selection.log", args.log_level)

    input_path = args.input or (root / "data/processed/nhanes_filtered_pre_imputation.csv")
    tables_dir = root / "outputs/tables"
    figures_dir = root / "outputs/figures"
    df = pd.read_csv(input_path)
    df = df.loc[df["cvd"].notna()].copy()
    X, y = split_features_target(df, target="cvd")
    original_features = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    logger.info("Running feature selection on rows=%s features=%s", X.shape[0], X.shape[1])

    method_tables = []
    methods: Dict[str, pd.Series] = {}
    methods["LASSO"] = lasso_importance(X_train, y_train, original_features, args.random_state)
    methods["Random Forest"] = rf_importance(X_train, y_train, original_features, args.random_state)
    methods["XGBoost"] = xgb_importance(X_train, y_train, original_features, args.random_state, logger)
    if not args.skip_boruta:
        methods["Boruta"] = boruta_importance(X_train, y_train, original_features, args.random_state, args.max_boruta_iter, logger)
    methods["Permutation"] = permutation_importance_scores(X_train, X_test, y_train, y_test, args.random_state)
    methods["SHAP"] = shap_importance_and_plot(
        X_train, X_test, y_train, y_test, figures_dir, original_features, args.random_state, args.shap_sample, logger
    )

    for method, importances in methods.items():
        table = rank_series(importances, method)
        if table.empty:
            logger.warning("%s produced no non-zero feature importance.", method)
        else:
            logger.info("%s top feature: %s", method, table.iloc[0]["feature"])
            method_tables.append(table)

    if not method_tables:
        raise RuntimeError("No feature selection method produced rankings.")

    long_table = pd.concat(method_tables, ignore_index=True)
    long_table.to_csv(tables_dir / "feature_importance_long.csv", index=False)
    consensus = build_consensus(long_table, args.top_n)
    consensus.to_csv(tables_dir / "table3_feature_ranking.csv", index=False)
    plot_consensus_bar(consensus, figures_dir)
    logger.info("Saved consensus feature ranking to %s", tables_dir / "table3_feature_ranking.csv")


if __name__ == "__main__":
    main()
