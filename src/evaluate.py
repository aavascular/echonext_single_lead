from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_binary_classification_metrics


def model_forward(model: torch.nn.Module, batch: dict[str, torch.Tensor], input_mode: str, device: torch.device) -> torch.Tensor:
    if input_mode == "tabular":
        return model(batch["tabular"].to(device))
    if input_mode in {"single_plus_tabular", "subset_plus_tabular"}:
        return model(batch["waveform"].to(device), batch["tabular"].to(device))
    return model(batch["waveform"].to(device))


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    input_mode: str,
    device: torch.device,
    desc: str = "Predict",
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(dataloader, desc=desc, leave=False):
        logits = model_forward(model, batch, input_mode, device)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()
        patient_keys = batch["patient_key"]
        row_indices = batch["row_index"].detach().cpu().numpy()
        for i in range(len(probabilities)):
            rows.append(
                {
                    "patient_key": patient_keys[i],
                    "row_index": int(row_indices[i]),
                    "label": float(labels[i]),
                    "logit": float(logits[i].detach().cpu().item()),
                    "probability": float(probabilities[i]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    y_true = predictions["label"].to_numpy().astype(int)
    y_prob = predictions["probability"].to_numpy().astype(float)
    return compute_binary_classification_metrics(y_true, y_prob)
