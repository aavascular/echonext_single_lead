from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


VALID_INPUT_MODES = {
    "full12",
    "single",
    "tabular",
    "single_plus_tabular",
    "subset",
    "subset_plus_tabular",
}
WAVEFORM_INPUT_MODES = {"full12", "single", "single_plus_tabular", "subset", "subset_plus_tabular"}


@dataclass
class SplitPaths:
    metadata_path: Path
    waveform_path: Path
    tabular_path: Path


def _normalize_split_name(split: str) -> str:
    return split.lower().strip()


def _read_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return pd.read_csv(metadata_path)


def resolve_split_paths(data_dir: str | Path, config: dict[str, Any], split: str) -> SplitPaths:
    split = _normalize_split_name(split)
    data_dir = Path(data_dir)
    return SplitPaths(
        metadata_path=data_dir / config["metadata_filename"],
        waveform_path=data_dir / config["waveform_files"][split],
        tabular_path=data_dir / config["tabular_files"][split],
    )


class EchoNextDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        config: dict[str, Any],
        split: str,
        label_col: str,
        input_mode: str,
        lead: str | None = None,
        leads: list[str] | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = _normalize_split_name(split)
        self.label_col = label_col
        self.input_mode = input_mode
        self.lead = lead
        self.leads = leads

        if self.input_mode not in VALID_INPUT_MODES:
            raise ValueError(f"Unsupported input_mode '{input_mode}'. Expected one of {sorted(VALID_INPUT_MODES)}.")
        if self.input_mode in {"single", "single_plus_tabular"} and not self.lead:
            raise ValueError("A lead name is required for single-lead input modes.")
        if self.input_mode in {"subset", "subset_plus_tabular"} and not self.leads:
            raise ValueError("A lead list is required for subset input modes.")

        self.lead_names = config["lead_names"]
        self.lead_to_idx = {name: idx for idx, name in enumerate(self.lead_names)}
        if self.lead and self.lead not in self.lead_to_idx:
            raise ValueError(f"Unknown lead '{self.lead}'. Expected one of {self.lead_names}.")
        if self.leads:
            unknown = [name for name in self.leads if name not in self.lead_to_idx]
            if unknown:
                raise ValueError(f"Unknown leads {unknown}. Expected values from {self.lead_names}.")

        split_paths = resolve_split_paths(self.data_dir, config, self.split)
        metadata = _read_metadata(split_paths.metadata_path)
        split_column = config["split_column"]
        patient_id_column = config["patient_id_column"]

        required_columns = [split_column, label_col, patient_id_column, *config["tabular_features"]]
        missing_columns = [column for column in required_columns if column not in metadata.columns]
        if missing_columns:
            raise KeyError(f"Metadata is missing required columns: {missing_columns}")

        split_mask = metadata[split_column].astype(str).str.lower() == self.split
        split_metadata = metadata.loc[split_mask].copy().reset_index().rename(columns={"index": "metadata_row_index"})
        split_metadata["split_array_index"] = np.arange(len(split_metadata))

        if split_metadata.empty:
            raise ValueError(f"No metadata rows found for split '{self.split}'.")

        label_mask = split_metadata[label_col].notna()
        split_metadata = split_metadata.loc[label_mask].copy().reset_index(drop=True)

        if split_metadata.empty:
            raise ValueError(f"No non-missing labels found for split '{self.split}' and label '{label_col}'.")

        self.waveforms = None
        if self.input_mode in WAVEFORM_INPUT_MODES:
            if not split_paths.waveform_path.exists():
                raise FileNotFoundError(f"Waveform file not found: {split_paths.waveform_path}")
            self.waveforms = np.load(split_paths.waveform_path, mmap_mode="r")

        if not split_paths.tabular_path.exists():
            raise FileNotFoundError(f"Tabular file not found: {split_paths.tabular_path}")
        self.tabular = np.load(split_paths.tabular_path, mmap_mode="r")

        split_size = int(split_mask.sum())
        if len(self.tabular) < split_size:
            raise ValueError(
                f"Tabular array length {len(self.tabular)} is smaller than split metadata length {split_size}."
            )
        if self.waveforms is not None and len(self.waveforms) < split_size:
            raise ValueError(
                f"Waveform array length {len(self.waveforms)} is smaller than split metadata length {split_size}."
            )

        self.split_metadata = split_metadata
        self.source_indices = split_metadata.index.to_numpy()
        if indices is None:
            self.dataset_indices = np.arange(len(split_metadata))
        else:
            self.dataset_indices = np.asarray(indices, dtype=int)

        self.patient_id_column = patient_id_column
        self.tabular_features = config["tabular_features"]

    def __len__(self) -> int:
        return len(self.dataset_indices)

    def _get_waveform(self, array_idx: int) -> torch.Tensor:
        assert self.waveforms is not None
        x = self.waveforms[array_idx]
        x = np.asarray(x).squeeze(0)
        if self.input_mode == "full12":
            x = x.T
        elif self.input_mode in {"subset", "subset_plus_tabular"}:
            lead_indices = [self.lead_to_idx[name] for name in self.leads or []]
            x = x[:, lead_indices].T
        else:
            lead_idx = self.lead_to_idx[self.lead]  # type: ignore[index]
            x = x[:, lead_idx][None, :]
        return torch.tensor(x, dtype=torch.float32)

    def _get_tabular(self, row: pd.Series, array_idx: int) -> torch.Tensor:
        if self.tabular is not None:
            tab = np.asarray(self.tabular[array_idx], dtype=np.float32)
        else:
            tab = row[self.tabular_features].to_numpy(dtype=np.float32)
        return torch.tensor(tab, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dataset_idx = int(self.dataset_indices[idx])
        row = self.split_metadata.iloc[dataset_idx]
        array_idx = int(row["split_array_index"])
        waveform = self._get_waveform(array_idx) if self.input_mode in WAVEFORM_INPUT_MODES else torch.empty(0)
        tabular = self._get_tabular(row, array_idx)
        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        return {
            "waveform": waveform,
            "tabular": tabular,
            "label": label,
            "patient_key": row[self.patient_id_column],
            "row_index": int(row["metadata_row_index"]),
        }

    def label_array(self) -> np.ndarray:
        return self.split_metadata.iloc[self.dataset_indices][self.label_col].astype(int).to_numpy()

    def prevalence(self) -> float:
        labels = self.label_array()
        return float(labels.mean()) if len(labels) else float("nan")


def subset_indices_stratified(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(labels))
    if fraction <= 0:
        raise ValueError("train_fraction must be greater than 0.")

    n_total = len(labels)
    n_target = max(1, int(round(n_total * fraction)))
    unique = np.unique(labels)
    stratify = labels if len(unique) > 1 and np.min(np.bincount(labels.astype(int))) >= 2 else None
    indices = np.arange(n_total)
    subset, _ = train_test_split(
        indices,
        train_size=n_target,
        random_state=seed,
        stratify=stratify,
    )
    return np.sort(subset)


def describe_split(dataset: EchoNextDataset) -> dict[str, Any]:
    labels = dataset.label_array()
    return {
        "split": dataset.split,
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else float("nan"),
    }
