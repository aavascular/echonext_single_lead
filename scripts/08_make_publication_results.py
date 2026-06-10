#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_binary_classification_metrics
from src.utils import ensure_dir, make_run_name


def core_experiments(include_six_limb: bool) -> list[dict[str, object]]:
    experiments: list[dict[str, object]] = [
        {"input_mode": "tabular", "lead": None, "leads": None, "model_name": "tabular only"},
        {"input_mode": "single", "lead": "I", "leads": None, "model_name": "lead I"},
        {"input_mode": "single_plus_tabular", "lead": "I", "leads": None, "model_name": "lead I + tabular"},
        {"input_mode": "full12", "lead": None, "leads": None, "model_name": "full 12-lead"},
    ]
    if include_six_limb:
        experiments.insert(
            3,
            {
                "input_mode": "subset",
                "lead": None,
                "leads": ["I", "II", "III", "aVR", "aVL", "aVF"],
                "model_name": "six limb leads",
            },
        )
    return experiments


def load_ensemble_predictions(
    predictions_dir: Path,
    label: str,
    experiment: dict[str, object],
    seeds: list[int],
) -> pd.DataFrame:
    seed_frames: list[pd.DataFrame] = []
    for seed in seeds:
        run_name = make_run_name(
            label=label,
            input_mode=str(experiment["input_mode"]),
            lead=experiment["lead"] if isinstance(experiment["lead"], str) else None,
            seed=seed,
            train_fraction=1.0,
            leads=experiment["leads"] if isinstance(experiment["leads"], list) else None,
        )
        prediction_path = predictions_dir / run_name / "test_predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing test predictions for required run: {prediction_path}")
        frame = pd.read_csv(prediction_path)
        frame = frame.rename(columns={"probability": f"probability_seed_{seed}", "logit": f"logit_seed_{seed}"})
        seed_frames.append(frame)

    merged = seed_frames[0]
    for frame in seed_frames[1:]:
        merged = merged.merge(
            frame[["patient_key", "row_index", "label", frame.columns[-2], frame.columns[-1]]],
            on=["patient_key", "row_index", "label"],
            how="inner",
            validate="one_to_one",
        )

    prob_cols = [column for column in merged.columns if column.startswith("probability_seed_")]
    merged["probability"] = merged[prob_cols].mean(axis=1)
    merged["model_name"] = str(experiment["model_name"])
    merged["input_mode"] = str(experiment["input_mode"])
    merged["lead"] = experiment["lead"]
    merged["leads"] = ",".join(experiment["leads"]) if isinstance(experiment["leads"], list) else None
    return merged


def bootstrap_metric_summary(
    predictions: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    point_metrics = compute_binary_classification_metrics(
        predictions["label"].to_numpy().astype(int),
        predictions["probability"].to_numpy().astype(float),
    )

    bootstrap_rows: list[dict[str, float]] = []
    n = len(predictions)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample = predictions.iloc[sample_idx]
        metrics = compute_binary_classification_metrics(
            sample["label"].to_numpy().astype(int),
            sample["probability"].to_numpy().astype(float),
        )
        bootstrap_rows.append(
            {
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier_score": metrics["brier_score"],
                "sensitivity_at_90_specificity": metrics["sensitivity_at_90_specificity"],
                "specificity_at_90_sensitivity": metrics["specificity_at_90_sensitivity"],
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    summary = {
        "n_test": int(point_metrics["n"]),
        "prevalence": float(point_metrics["prevalence"]),
        "AUROC": float(point_metrics["auroc"]),
        "AUPRC": float(point_metrics["auprc"]),
        "Brier score": float(point_metrics["brier_score"]),
        "sensitivity_at_90_specificity": float(point_metrics["sensitivity_at_90_specificity"]),
        "specificity_at_90_sensitivity": float(point_metrics["specificity_at_90_sensitivity"]),
    }
    for metric_name, output_name in [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("brier_score", "Brier score"),
        ("sensitivity_at_90_specificity", "sensitivity_at_90_specificity"),
        ("specificity_at_90_sensitivity", "specificity_at_90_sensitivity"),
    ]:
        summary[f"{output_name}_ci_lower"] = float(bootstrap_df[metric_name].quantile(0.025))
        summary[f"{output_name}_ci_upper"] = float(bootstrap_df[metric_name].quantile(0.975))
    return summary


def plot_roc_pr(ensemble_predictions: list[pd.DataFrame], figures_dir: Path) -> None:
    plt.figure(figsize=(7, 6))
    for predictions in ensemble_predictions:
        y_true = predictions["label"].to_numpy().astype(int)
        y_prob = predictions["probability"].to_numpy().astype(float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=predictions["model_name"].iloc[0])
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for core models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "publication_core_roc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    for predictions in ensemble_predictions:
        y_true = predictions["label"].to_numpy().astype(int)
        y_prob = predictions["probability"].to_numpy().astype(float)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, label=predictions["model_name"].iloc[0])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves for core models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "publication_core_pr.png", dpi=200)
    plt.close()


def plot_bar_with_ci(summary_df: pd.DataFrame, metric: str, figures_dir: Path, filename: str) -> None:
    ordered = summary_df.sort_values(metric, ascending=False)
    labels = ordered["model_name"].tolist()
    values = ordered[metric].to_numpy()
    lower = ordered[f"{metric}_ci_lower"].to_numpy()
    upper = ordered[f"{metric}_ci_upper"].to_numpy()
    yerr = np.vstack([values - lower, upper - values])

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, alpha=0.8)
    plt.errorbar(labels, values, yerr=yerr, fmt="none", ecolor="black", capsize=4)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} with 95% bootstrap confidence intervals")
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build publication-style core result tables and figures.")
    parser.add_argument("--label", type=str, default="shd_moderate_or_greater_flag")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include_six_limb", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")
    rng = np.random.default_rng(args.seed)

    experiments = core_experiments(include_six_limb=args.include_six_limb)
    ensemble_predictions: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for experiment in experiments:
        predictions = load_ensemble_predictions(
            predictions_dir=predictions_dir,
            label=args.label,
            experiment=experiment,
            seeds=args.seeds,
        )
        ensemble_predictions.append(predictions)
        metrics_summary = bootstrap_metric_summary(predictions, rng, args.bootstrap_iterations)
        summary_rows.append(
            {
                "label": args.label,
                "input_mode": experiment["input_mode"],
                "lead": experiment["lead"],
                "leads": ",".join(experiment["leads"]) if isinstance(experiment["leads"], list) else None,
                "model_name": experiment["model_name"],
                "n_seeds": len(args.seeds),
                **metrics_summary,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("AUROC", ascending=False)
    summary_df.to_csv(tables_dir / "publication_core_results.csv", index=False)

    plot_roc_pr(ensemble_predictions, figures_dir)
    plot_bar_with_ci(summary_df, "AUROC", figures_dir, "publication_core_auroc_ci.png")
    plot_bar_with_ci(summary_df, "AUPRC", figures_dir, "publication_core_auprc_ci.png")

    print(f"Saved publication summary to {tables_dir / 'publication_core_results.csv'}")
    print(f"Saved publication figures to {figures_dir}")


if __name__ == "__main__":
    main()
