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

from src.train import run_training
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label-efficiency experiments.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label", type=str, default="shd_moderate_or_greater_flag")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    fractions = [0.005, 0.01, 0.05, 0.20, 0.50, 1.00]
    variants = [("single", "I"), ("full12", None)]
    summaries: list[dict] = []

    for input_mode, lead in variants:
        for fraction in fractions:
            run_args = Namespace(
                data_dir=args.data_dir,
                label=args.label,
                input_mode=input_mode,
                lead=lead,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                output_dir=args.output_dir,
                train_fraction=fraction,
                config=args.config,
            )
            metrics = run_training(run_args)
            summaries.append(
                {
                    "label": args.label,
                    "input_mode": input_mode,
                    "lead": lead,
                    "train_fraction": fraction,
                    "n_train": metrics["splits"]["train"]["n"],
                    "AUROC": metrics["test_metrics"]["auroc"],
                    "AUPRC": metrics["test_metrics"]["auprc"],
                    "sensitivity_at_90_specificity": metrics["test_metrics"]["sensitivity_at_90_specificity"],
                    "specificity_at_90_sensitivity": metrics["test_metrics"]["specificity_at_90_sensitivity"],
                    "Brier score": metrics["test_metrics"]["brier_score"],
                }
            )

    output_path = ensure_dir(args.output_dir) / "tables"
    ensure_dir(output_path)
    pd.DataFrame(summaries).to_csv(output_path / "label_efficiency_results.csv", index=False)


if __name__ == "__main__":
    main()
