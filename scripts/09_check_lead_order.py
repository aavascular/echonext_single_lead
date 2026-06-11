#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import ensure_dir, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanity-check assumed ECG lead order.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n_examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser


def load_waveforms(data_dir: Path, config: dict, split: str) -> np.memmap:
    waveform_path = data_dir / config["waveform_files"][split]
    if not waveform_path.exists():
        raise FileNotFoundError(f"Waveform file not found: {waveform_path}")
    return np.load(waveform_path, mmap_mode="r")


def sample_indices(n_total: int, n_examples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_examples = min(n_examples, n_total)
    return np.sort(rng.choice(n_total, size=n_examples, replace=False))


def flatten_leads(waveforms: np.ndarray, lead_names: list[str]) -> pd.DataFrame:
    records: dict[str, np.ndarray] = {}
    for lead_idx, lead_name in enumerate(lead_names):
        records[lead_name] = waveforms[:, lead_idx, :].reshape(-1)
    return pd.DataFrame(records)


def relative_error(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(np.linalg.norm(numerator) / (np.linalg.norm(denominator) + 1e-8))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    output_root = ensure_dir(Path(args.output_dir) / "lead_order_checks")

    lead_names = config["lead_names"]
    waveforms_mm = load_waveforms(data_dir, config, args.split)
    indices = sample_indices(len(waveforms_mm), args.n_examples, args.seed)

    sampled = np.asarray(waveforms_mm[indices], dtype=np.float32)  # shape: N x 1 x 2500 x 12
    sampled = sampled.squeeze(1)  # N x 2500 x 12
    sampled = np.transpose(sampled, (0, 2, 1))  # N x 12 x 2500

    waveform_df = flatten_leads(sampled, lead_names)
    corr = waveform_df.corr()

    lead_to_idx = {lead: idx for idx, lead in enumerate(lead_names)}
    lead_i = sampled[:, lead_to_idx["I"], :]
    lead_ii = sampled[:, lead_to_idx["II"], :]
    lead_iii = sampled[:, lead_to_idx["III"], :]
    lead_avr = sampled[:, lead_to_idx["aVR"], :]
    lead_avl = sampled[:, lead_to_idx["aVL"], :]
    lead_avf = sampled[:, lead_to_idx["aVF"], :]

    einthoven_residual = lead_ii - (lead_i + lead_iii)
    augmented_sum = lead_avr + lead_avl + lead_avf

    einthoven_rel_error = relative_error(einthoven_residual.reshape(-1), lead_ii.reshape(-1))
    augmented_rel_error = relative_error(augmented_sum.reshape(-1), lead_avr.reshape(-1))

    per_example_einthoven = np.linalg.norm(einthoven_residual.reshape(len(sampled), -1), axis=1) / (
        np.linalg.norm(lead_ii.reshape(len(sampled), -1), axis=1) + 1e-8
    )
    per_example_augmented = np.linalg.norm(augmented_sum.reshape(len(sampled), -1), axis=1) / (
        np.linalg.norm(lead_avr.reshape(len(sampled), -1), axis=1) + 1e-8
    )

    summary = {
        "split": args.split,
        "n_examples": int(len(indices)),
        "assumed_lead_order": lead_names,
        "einthoven_relative_error_mean": float(np.mean(per_example_einthoven)),
        "einthoven_relative_error_median": float(np.median(per_example_einthoven)),
        "einthoven_relative_error_global": einthoven_rel_error,
        "augmented_relative_error_mean": float(np.mean(per_example_augmented)),
        "augmented_relative_error_median": float(np.median(per_example_augmented)),
        "augmented_relative_error_global": augmented_rel_error,
        "interpretation_note": (
            "Smaller residuals are more consistent with the assumed standard lead ordering. "
            "These checks are heuristic because dataset preprocessing may alter exact algebraic identities."
        ),
    }

    save_json(summary, output_root / f"{args.split}_lead_order_summary.json")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, square=True)
    plt.title(f"Lead correlation heatmap ({args.split}, n={len(indices)})")
    plt.tight_layout()
    plt.savefig(output_root / f"{args.split}_lead_correlation_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(per_example_einthoven, bins=40, alpha=0.8)
    plt.xlabel("Relative error")
    plt.ylabel("Count")
    plt.title("Einthoven residual distribution: II - (I + III)")
    plt.tight_layout()
    plt.savefig(output_root / f"{args.split}_einthoven_residual_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(per_example_augmented, bins=40, alpha=0.8)
    plt.xlabel("Relative error")
    plt.ylabel("Count")
    plt.title("Augmented lead residual distribution: aVR + aVL + aVF")
    plt.tight_layout()
    plt.savefig(output_root / f"{args.split}_augmented_residual_hist.png", dpi=200)
    plt.close()

    print("Assumed lead order:")
    print(lead_names)
    print()
    print(f"Split: {args.split}")
    print(f"Sampled ECGs: {len(indices)}")
    print(f"Einthoven relative error (global): {einthoven_rel_error:.4f}")
    print(f"Einthoven relative error (mean per ECG): {np.mean(per_example_einthoven):.4f}")
    print(f"Augmented lead relative error (global): {augmented_rel_error:.4f}")
    print(f"Augmented lead relative error (mean per ECG): {np.mean(per_example_augmented):.4f}")
    print()
    print(f"Saved summary to: {output_root / f'{args.split}_lead_order_summary.json'}")
    print(f"Saved figures to: {output_root}")


if __name__ == "__main__":
    main()
