import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402
from cvd_nhanes_gbd.model_utils import (  # noqa: E402
    build_model_configs,
    evaluate_binary_classifier,
    infer_feature_types,
    make_model_pipeline,
    make_preprocessor,
    optional_package_available,
    predict_probability,
    roc_pr_points,
)
from cvd_nhanes_gbd.paths import ensure_project_dirs  # noqa: E402
from cvd_nhanes_gbd.model_utils import split_features_target  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate CVD prediction models from NHANES data.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input", type=Path, default=None, help="Default: data/processed/nhanes_filtered_pre_imputation.csv")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE inside the training pipeline only.")
    parser.add_argument("--models", default="Logistic Regression,Random Forest,XGBoost,LightGBM,SVM,Elastic Net")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_pipeline(name: str, estimator: object, X_train: pd.DataFrame, use_smote: bool, random_state: int, logger):
    if not use_smote:
        return make_model_pipeline(name, estimator, X_train)
    if not optional_package_available("imblearn"):
        logger.warning("imblearn is not installed; SMOTE disabled for %s.", name)
        return make_model_pipeline(name, estimator, X_train)
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    scale = name in {"Logistic Regression", "Elastic Net", "SVM"}
    preprocessor = make_preprocessor(X_train, scale_numeric=scale)
    return ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", estimator),
        ]
    )


def plot_combined_curves(results, y_test, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for result in results:
        name = result["model"]
        y_prob = result["y_prob"]
        (fpr, tpr), (recall, precision) = roc_pr_points(y_test, y_prob)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} AUC={result['auc']:.3f}")
        axes[1].plot(recall, precision, lw=2, label=f"{name} PR-AUC={result['pr_auc']:.3f}")
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
        axes[2].plot(prob_pred, prob_true, marker="o", lw=1.8, label=name)

    axes[0].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall")
    axes[1].legend(fontsize=8)

    axes[2].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Observed probability")
    axes[2].set_title("Calibration")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_roc_pr_calibration.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrices(results, figures_dir: Path) -> None:
    n = len(results)
    if n == 0:
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, result in zip(axes, results):
        mat = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(result["model"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_xticklabels(["No CVD", "CVD"])
        ax.set_yticklabels(["No CVD", "CVD"], rotation=0)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_project_dirs(root)
    logger = setup_logging("ml_modeling", root / "outputs/logs/ml_modeling.log", args.log_level)

    input_path = args.input or (root / "data/processed/nhanes_filtered_pre_imputation.csv")
    tables_dir = root / "outputs/tables"
    figures_dir = root / "outputs/figures"
    models_dir = root / "outputs/models"

    df = pd.read_csv(input_path)
    df = df.loc[df["cvd"].notna()].copy()
    X, y = split_features_target(df, target="cvd")
    if y.nunique() < 2:
        raise RuntimeError("The CVD outcome has fewer than two classes; model training cannot proceed.")
    logger.info("Model matrix before preprocessing: rows=%s features=%s", X.shape[0], X.shape[1])
    numeric_cols, categorical_cols = infer_feature_types(X)
    logger.info("Numeric features=%s; categorical features=%s", len(numeric_cols), len(categorical_cols))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    configs = build_model_configs(y_train, args.random_state, logger)
    configs = {name: model for name, model in configs.items() if name in requested}
    if not configs:
        raise RuntimeError("No requested models are available.")

    records = []
    curve_results = []
    prediction_frames = []
    best_auc = -np.inf
    best_name = None
    best_model = None

    for name, estimator in configs.items():
        logger.info("Training %s", name)
        pipeline = build_pipeline(name, estimator, X_train, args.smote, args.random_state, logger)
        try:
            pipeline.fit(X_train, y_train)
            y_prob = predict_probability(pipeline, X_test)
            metrics = evaluate_binary_classifier(y_test, y_prob)
        except Exception as exc:
            logger.exception("%s failed and will be skipped: %s", name, exc)
            continue
        row = {"model": name, "n_train": len(y_train), "n_test": len(y_test), "smote": args.smote}
        row.update(metrics)
        records.append(row)
        curve_results.append({"model": name, "y_prob": y_prob, **metrics})
        prediction_frames.append(
            pd.DataFrame({"model": name, "observed_cvd": y_test.to_numpy(), "predicted_probability": y_prob})
        )
        joblib.dump(pipeline, models_dir / f"{name.lower().replace(' ', '_')}.joblib")
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_name = name
            best_model = pipeline
        logger.info("%s AUC=%.3f PR-AUC=%.3f Brier=%.3f", name, metrics["auc"], metrics["pr_auc"], metrics["brier_score"])

    if not records:
        raise RuntimeError("All model fits failed; inspect outputs/logs/ml_modeling.log.")

    performance = pd.DataFrame(records).sort_values("auc", ascending=False)
    performance.to_csv(tables_dir / "table2_model_performance.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(tables_dir / "model_test_predictions.csv", index=False)
    if best_model is not None:
        joblib.dump(best_model, models_dir / "best_model.joblib")
        logger.info("Best model by AUC: %s", best_name)

    plot_combined_curves(curve_results, y_test, figures_dir)
    plot_confusion_matrices(curve_results, figures_dir)
    logger.info("Saved model performance table and figures.")


if __name__ == "__main__":
    main()
