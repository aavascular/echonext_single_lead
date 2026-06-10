#!/usr/bin/env python
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.train import run_training
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lead sweep benchmark.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    labels = [args.label] if args.label else config["initial_labels"]
    leads = config["lead_names"]
    summaries: list[dict] = []

    experiments = []
    for label in labels:
        experiments.append(("tabular", None, label))
        experiments.append(("full12", None, label))
        experiments.append(("single_plus_tabular", "I", label))
        experiments.append(("subset", None, label))
        for lead in leads:
            experiments.append(("single", lead, label))

    for input_mode, lead, label in experiments:
        leads_arg = "I,II,III,aVR,aVL,aVF" if input_mode == "subset" else None
        run_args = Namespace(
            data_dir=args.data_dir,
            label=label,
            input_mode=input_mode,
            lead=lead,
            leads=leads_arg,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            output_dir=args.output_dir,
            train_fraction=1.0,
            config=args.config,
        )
        metrics = run_training(run_args)
        row = {
            "label": label,
            "input_mode": input_mode,
            "lead": lead,
            "leads": leads_arg,
            "n_test": metrics["test_metrics"]["n"],
            "prevalence": metrics["test_metrics"]["prevalence"],
            "AUROC": metrics["test_metrics"]["auroc"],
            "AUPRC": metrics["test_metrics"]["auprc"],
            "sensitivity_at_90_specificity": metrics["test_metrics"]["sensitivity_at_90_specificity"],
            "specificity_at_90_sensitivity": metrics["test_metrics"]["specificity_at_90_sensitivity"],
            "Brier score": metrics["test_metrics"]["brier_score"],
        }
        summaries.append(row)

    output_path = ensure_dir(args.output_dir) / "tables"
    ensure_dir(output_path)
    pd.DataFrame(summaries).to_csv(output_path / "model_performance_by_input.csv", index=False)


if __name__ == "__main__":
    main()
