#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir


def load_run_metrics(models_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in models_dir.glob("*/metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        row = {
            "label": metrics["label"],
            "input_mode": metrics["input_mode"],
            "lead": metrics["lead"],
            "leads": metrics.get("leads"),
            "train_fraction": metrics["train_fraction"],
            "n_train": metrics["splits"]["train"]["n"],
            "n_test": metrics["test_metrics"]["n"],
            "prevalence": metrics["test_metrics"]["prevalence"],
            "AUROC": metrics["test_metrics"]["auroc"],
            "AUPRC": metrics["test_metrics"]["auprc"],
            "sensitivity_at_90_specificity": metrics["test_metrics"]["sensitivity_at_90_specificity"],
            "specificity_at_90_sensitivity": metrics["test_metrics"]["specificity_at_90_sensitivity"],
            "Brier score": metrics["test_metrics"]["brier_score"],
            "calibration_curve": metrics["test_metrics"]["calibration_curve"],
        }
        rows.append(row)
    return rows


def calibration_label(row: pd.Series) -> str:
    if row["input_mode"] == "tabular":
        return "tabular only"
    if row["input_mode"] == "full12":
        return "full 12-lead"
    if row["input_mode"] == "subset":
        return "six limb leads"
    if row["input_mode"] == "single_plus_tabular":
        return "lead I + tabular"
    return f"lead {row['lead']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics and build figures.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")

    rows = load_run_metrics(models_dir)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found in {models_dir}")

    df = pd.DataFrame(rows)

    benchmark_df = df[df["train_fraction"] == 1.0].copy()
    benchmark_table_df = benchmark_df[
        [
            "label",
            "input_mode",
            "lead",
            "n_test",
            "prevalence",
            "AUROC",
            "AUPRC",
            "sensitivity_at_90_specificity",
            "specificity_at_90_sensitivity",
            "Brier score",
        ]
    ]
    benchmark_table_df.to_csv(tables_dir / "model_performance_by_input.csv", index=False)

    efficiency_df = df[df["train_fraction"] != 1.0].copy()
    if not efficiency_df.empty:
        efficiency_df = efficiency_df[
            [
                "label",
                "input_mode",
                "lead",
                "train_fraction",
                "n_train",
                "AUROC",
                "AUPRC",
                "sensitivity_at_90_specificity",
                "specificity_at_90_sensitivity",
                "Brier score",
            ]
        ]
        efficiency_df.to_csv(tables_dir / "label_efficiency_results.csv", index=False)

    if not benchmark_df.empty:
        auroc_plot_df = benchmark_df.copy()
        auroc_plot_df["label_name"] = auroc_plot_df.apply(calibration_label, axis=1)
        plt.figure(figsize=(12, 6))
        for label, sub_df in auroc_plot_df.groupby("label"):
            plt.bar(sub_df["label_name"], sub_df["AUROC"], alpha=0.6, label=label)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("AUROC")
        plt.title("AUROC by input type")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "auroc_by_input.png", dpi=200)
        plt.close()

        plt.figure(figsize=(12, 6))
        for label, sub_df in auroc_plot_df.groupby("label"):
            plt.bar(sub_df["label_name"], sub_df["AUPRC"], alpha=0.6, label=label)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("AUPRC")
        plt.title("AUPRC by input type")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "auprc_by_input.png", dpi=200)
        plt.close()

        calib_df = benchmark_df[
            (benchmark_df["input_mode"] == "tabular")
            | (benchmark_df["input_mode"] == "full12")
            | ((benchmark_df["input_mode"] == "single") & (benchmark_df["lead"] == "I"))
        ].copy()
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="black")
        for _, row in calib_df.iterrows():
            curve = row["calibration_curve"]
            plt.plot(curve["predicted"], curve["observed"], marker="o", label=calibration_label(row))
        plt.xlabel("Mean predicted risk")
        plt.ylabel("Observed prevalence")
        plt.title("Calibration plot")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "calibration_plot.png", dpi=200)
        plt.close()

    if not efficiency_df.empty:
        plt.figure(figsize=(8, 6))
        for (input_mode, lead), sub_df in efficiency_df.groupby(["input_mode", "lead"], dropna=False):
            label = "full12" if input_mode == "full12" else f"lead {lead}"
            sub_df = sub_df.sort_values("train_fraction")
            plt.plot(sub_df["train_fraction"], sub_df["AUROC"], marker="o", label=label)
        plt.xscale("log")
        plt.xlabel("Training fraction")
        plt.ylabel("AUROC")
        plt.title("Label-efficiency AUROC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "label_efficiency_auroc.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
