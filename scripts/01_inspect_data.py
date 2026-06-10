#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect EchoNext metadata and arrays.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / config["metadata_filename"]
    metadata = pd.read_csv(metadata_path)

    print(f"Metadata path: {metadata_path}")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Columns: {list(metadata.columns)}")

    split_column = config["split_column"]
    for split in ["train", "val", "test", "no_split"]:
        split_rows = metadata[metadata[split_column].astype(str).str.lower() == split]
        print(f"\nSplit: {split}")
        print(f"Metadata rows: {len(split_rows)}")

        waveform_path = data_dir / config["waveform_files"][split]
        tabular_path = data_dir / config["tabular_files"][split]

        if waveform_path.exists():
            waveform_array = np.load(waveform_path, mmap_mode="r")
            print(f"Waveforms: {waveform_array.shape}")
        else:
            print(f"Waveforms: missing ({waveform_path.name})")

        if tabular_path.exists():
            tabular_array = np.load(tabular_path, mmap_mode="r")
            print(f"Tabular: {tabular_array.shape}")
        else:
            print(f"Tabular: missing ({tabular_path.name})")

        for label in config["initial_labels"]:
            if label in split_rows.columns:
                prevalence = split_rows[label].dropna().mean()
                observed = split_rows[label].notna().sum()
                print(f"Label {label}: observed={observed} prevalence={prevalence:.4f}" if observed else f"Label {label}: observed=0")


if __name__ == "__main__":
    main()
