#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular baseline.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--train_fraction", type=float, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    args.input_mode = "tabular"
    args.lead = None
    run_training(args)


if __name__ == "__main__":
    main()
