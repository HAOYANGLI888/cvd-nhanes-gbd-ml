from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvd_nhanes_gbd.feature_sets import sanitized_features  # noqa: E402
from cvd_nhanes_gbd.logging_utils import setup_logging  # noqa: E402
from cvd_nhanes_gbd.model_utils import build_model_configs, evaluate_binary_classifier, make_model_pipeline, predict_probability  # noqa: E402
from cvd_nhanes_gbd.preprocessing import normalize_missing_values  # noqa: E402


FINAL_MODELS = {
    "full_clinical_model": "XGBoost",
    "routine_indicator_model": "XGBoost",
    "biomarker_focused_model": "Elastic Net",
}

INTERNAL_AUC = {
    "full_clinical_model": 0.8528,
    "routine_indicator_model": 0.8478,
    "biomarker_focused_model": 0.8460,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External temporal validation of 2005-2018 NHANES final models on 2021-2023.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--train-data", type=Path, default=None)
    parser.add_argument("--external-data", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def align_external_features(X_train: pd.DataFrame, external: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    missing_features = [feature for feature in features if feature not in external.columns]
    aligned = external.copy()
    for feature in missing_features:
        aligned[feature] = np.nan
    aligned = aligned[features].copy()
    for feature in features:
        if pd.api.types.is_numeric_dtype(X_train[feature]):
            aligned[feature] = pd.to_numeric(aligned[feature], errors="coerce")
        else:
            aligned[feature] = aligned[feature].astype("object")
    return aligned, missing_features


def calibration_intercept_slope(y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, float]:
    if y_true.nunique(dropna=True) < 2:
        return np.nan, np.nan
    p = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        try:
            model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        except TypeError:
            model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
        model.fit(logit, y_true.astype(int))
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return np.nan, np.nan


def plot_external_curves(prediction_rows: List[Dict], y_true: pd.Series, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for row in prediction_rows:
        y_prob = row["y_prob"]
        label = f"{row['feature_set']} ({row['model']})"
        if y_true.nunique() >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            axes[0].plot(fpr, tpr, lw=2, label=f"{label} AUC={row['auc']:.3f}")
            axes[1].plot(recall, precision, lw=2, label=f"{label} PR-AUC={row['pr_auc']:.3f}")
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
            axes[2].plot(prob_pred, prob_true, marker="o", lw=1.8, label=label)
        except Exception:
            continue
    axes[0].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("External ROC")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("External precision-recall")
    axes[1].legend(fontsize=7)
    axes[2].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Observed probability")
    axes[2].set_title("External calibration")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = root / "outputs/external_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("external_validate_2021_2023", root / "outputs/logs/external_validate_2021_2023.log", args.log_level)
    train_path = args.train_data or (root / "data/processed/nhanes_filtered_pre_imputation.csv")
    external_path = args.external_data or (root / "data/processed/nhanes_2021_2023_external.csv")

    train = normalize_missing_values(pd.read_csv(train_path))
    external = normalize_missing_values(pd.read_csv(external_path))
    train = train.loc[train["cvd"].notna()].copy()
    external = external.loc[external["cvd"].notna()].copy()
    y_train = pd.to_numeric(train["cvd"], errors="coerce").astype(int)
    y_external = pd.to_numeric(external["cvd"], errors="coerce").astype(int)

    rows = []
    prediction_rows = []
    missing_feature_rows = []
    for feature_set, model_name in FINAL_MODELS.items():
        features = sanitized_features(train, feature_set)
        X_train = train[features].copy()
        X_external, missing_features = align_external_features(X_train, external, features)
        for feature in missing_features:
            missing_feature_rows.append({"feature_set": feature_set, "missing_feature_in_external": feature})
        configs = build_model_configs(y_train, args.random_state, logger)
        if model_name not in configs:
            logger.warning("Final model %s unavailable for %s; skipping.", model_name, feature_set)
            continue
        pipeline = make_model_pipeline(model_name, configs[model_name], X_train)
        pipeline.fit(X_train, y_train)
        model_path = output_dir / f"final_{feature_set}_{model_name.replace(' ', '_').lower()}_trained_2005_2018.joblib"
        joblib.dump(pipeline, model_path)
        y_prob = predict_probability(pipeline, X_external)
        metrics = evaluate_binary_classifier(y_external, y_prob)
        intercept, slope = calibration_intercept_slope(y_external, y_prob)
        row = {
            "feature_set": feature_set,
            "final_model": model_name,
            "train_source": "NHANES 2005-2018 full Stage 2 analytic dataset",
            "external_source": "NHANES 2021-2023 fasting-core external cohort",
            "n_train": len(y_train),
            "n_external": len(y_external),
            "n_external_events": int((y_external == 1).sum()),
            "n_external_nonevents": int((y_external == 0).sum()),
            "AUC": metrics["auc"],
            "PR-AUC": metrics["pr_auc"],
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "F1": metrics["f1"],
            "Brier score": metrics["brier_score"],
            "calibration intercept": intercept,
            "calibration slope": slope,
            "internal held-out AUC": INTERNAL_AUC.get(feature_set, np.nan),
            "external minus internal AUC": metrics["auc"] - INTERNAL_AUC.get(feature_set, np.nan),
            "n_missing_features_in_external": len(missing_features),
            "model_path": str(model_path),
        }
        rows.append(row)
        prediction_rows.append({"feature_set": feature_set, "model": model_name, "y_prob": y_prob, "auc": metrics["auc"], "pr_auc": metrics["pr_auc"]})
        pd.DataFrame(
            {
                "seqn": external.get("seqn", pd.Series(np.arange(len(external)))),
                "cvd": y_external,
                "feature_set": feature_set,
                "model": model_name,
                "predicted_probability": y_prob,
            }
        ).to_csv(output_dir / f"external_predictions_{feature_set}.csv", index=False)
    performance = pd.DataFrame(rows)
    performance.to_csv(output_dir / "external_model_performance_2021_2023.csv", index=False)
    pd.DataFrame(missing_feature_rows).to_csv(output_dir / "external_model_missing_features.csv", index=False)
    if prediction_rows:
        plot_external_curves(prediction_rows, y_external, output_dir / "external_roc_pr_calibration.png")
    logger.info("External validation complete: %s model rows", len(rows))


if __name__ == "__main__":
    main()
