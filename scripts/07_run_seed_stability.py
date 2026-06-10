#!/usr/bin/env python
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import run_training
from src.utils import ensure_dir


def display_name(input_mode: str, lead: str | None) -> str:
    if input_mode == "tabular":
        return "tabular only"
    if input_mode == "full12":
        return "full 12-lead"
    if input_mode == "single_plus_tabular":
        return "lead I + tabular"
    return f"lead {lead}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated-seed stability experiments for core models.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label", type=str, default="shd_moderate_or_greater_flag")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--config", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    experiments = [
        ("tabular", None),
        ("single", "I"),
        ("single_plus_tabular", "I"),
        ("full12", None),
    ]

    rows: list[dict] = []
    for seed in args.seeds:
        for input_mode, lead in experiments:
            run_args = Namespace(
                data_dir=args.data_dir,
                label=args.label,
                input_mode=input_mode,
                lead=lead,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                seed=seed,
                output_dir=args.output_dir,
                train_fraction=1.0,
                config=args.config,
            )
            metrics = run_training(run_args)
            rows.append(
                {
                    "label": args.label,
                    "seed": seed,
                    "input_mode": input_mode,
                    "lead": lead,
                    "model_name": display_name(input_mode, lead),
                    "n_test": metrics["test_metrics"]["n"],
                    "prevalence": metrics["test_metrics"]["prevalence"],
                    "AUROC": metrics["test_metrics"]["auroc"],
                    "AUPRC": metrics["test_metrics"]["auprc"],
                    "sensitivity_at_90_specificity": metrics["test_metrics"]["sensitivity_at_90_specificity"],
                    "specificity_at_90_sensitivity": metrics["test_metrics"]["specificity_at_90_sensitivity"],
                    "Brier score": metrics["test_metrics"]["brier_score"],
                }
            )

    results_df = pd.DataFrame(rows)
    tables_dir = ensure_dir(Path(args.output_dir) / "tables")
    figures_dir = ensure_dir(Path(args.output_dir) / "figures")

    raw_path = tables_dir / "seed_stability_results.csv"
    results_df.to_csv(raw_path, index=False)

    summary_df = (
        results_df.groupby(["label", "input_mode", "lead", "model_name"], dropna=False)
        .agg(
            n_runs=("seed", "count"),
            AUROC_mean=("AUROC", "mean"),
            AUROC_std=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_std=("AUPRC", "std"),
            Brier_mean=("Brier score", "mean"),
            Brier_std=("Brier score", "std"),
            sensitivity_at_90_specificity_mean=("sensitivity_at_90_specificity", "mean"),
            specificity_at_90_sensitivity_mean=("specificity_at_90_sensitivity", "mean"),
        )
        .reset_index()
        .sort_values("AUROC_mean", ascending=False)
    )
    summary_path = tables_dir / "seed_stability_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    ordered_models = summary_df["model_name"].tolist()
    x_positions = {name: idx for idx, name in enumerate(ordered_models)}

    plt.figure(figsize=(10, 6))
    for model_name, sub_df in results_df.groupby("model_name"):
        x = [x_positions[model_name]] * len(sub_df)
        plt.scatter(x, sub_df["AUROC"], alpha=0.8, s=60, label=model_name)

    for _, row in summary_df.iterrows():
        x = x_positions[row["model_name"]]
        plt.errorbar(
            x,
            row["AUROC_mean"],
            yerr=0.0 if pd.isna(row["AUROC_std"]) else row["AUROC_std"],
            fmt="_",
            color="black",
            capsize=4,
            markersize=18,
            linewidth=1.5,
        )

    plt.xticks(list(x_positions.values()), list(x_positions.keys()), rotation=25, ha="right")
    plt.ylabel("Test AUROC")
    plt.title("Seed stability across core models")
    plt.tight_layout()
    plt.savefig(figures_dir / "seed_stability_auroc.png", dpi=200)
    plt.close()

    print(f"Saved raw seed results to {raw_path}")
    print(f"Saved seed summary to {summary_path}")
    print(f"Saved AUROC stability figure to {figures_dir / 'seed_stability_auroc.png'}")


if __name__ == "__main__":
    main()
