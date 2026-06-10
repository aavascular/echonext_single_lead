from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data import EchoNextDataset, describe_split, subset_indices_stratified
from src.evaluate import evaluate_predictions, predict, model_forward
from src.models import build_model
from src.utils import ensure_dir, get_device, make_run_name, parse_leads, save_json, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EchoNext models.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument(
        "--input_mode",
        type=str,
        required=True,
        choices=["full12", "single", "tabular", "single_plus_tabular", "subset", "subset_plus_tabular"],
    )
    parser.add_argument("--lead", type=str, default=None)
    parser.add_argument("--leads", type=str, default=None, help="Comma-separated lead names for subset modes.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--train_fraction", type=float, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser


def make_dataloaders(args: argparse.Namespace, config: dict[str, Any]) -> tuple[dict[str, EchoNextDataset], dict[str, DataLoader], dict[str, Any]]:
    lead_list = parse_leads(getattr(args, "leads", None))
    train_dataset = EchoNextDataset(
        data_dir=args.data_dir,
        config=config,
        split="train",
        label_col=args.label,
        input_mode=args.input_mode,
        lead=args.lead,
        leads=lead_list,
    )

    train_fraction = args.train_fraction if args.train_fraction is not None else config["training"]["train_fraction"]
    train_indices = subset_indices_stratified(train_dataset.label_array(), float(train_fraction), seed=args.seed)
    if len(train_indices) != len(train_dataset):
        train_dataset = EchoNextDataset(
            data_dir=args.data_dir,
            config=config,
            split="train",
            label_col=args.label,
            input_mode=args.input_mode,
            lead=args.lead,
            leads=lead_list,
            indices=train_indices,
        )

    val_dataset = EchoNextDataset(
        data_dir=args.data_dir,
        config=config,
        split="val",
        label_col=args.label,
        input_mode=args.input_mode,
        lead=args.lead,
        leads=lead_list,
    )
    test_dataset = EchoNextDataset(
        data_dir=args.data_dir,
        config=config,
        split="test",
        label_col=args.label,
        input_mode=args.input_mode,
        lead=args.lead,
        leads=lead_list,
    )

    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    info = {
        "train_fraction": float(train_fraction),
        "batch_size": int(batch_size),
    }
    return datasets, dataloaders, info


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    if positives <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(max(negatives / positives, 1.0), dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    input_mode: str,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad()
        logits = model_forward(model, batch, input_mode, device)
        labels = batch["label"].to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def select_improved(metric_name: str, current_best: float | None, candidate: float) -> bool:
    if current_best is None:
        return True
    if metric_name == "val_loss":
        return candidate < current_best
    return candidate > current_best


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    lead_list = parse_leads(getattr(args, "leads", None))

    datasets, dataloaders, loader_info = make_dataloaders(args, config)
    splits = {name: describe_split(dataset) for name, dataset in datasets.items()}

    for split_name, stats in splits.items():
        prevalence = stats["prevalence"]
        if prevalence == prevalence:
            print(f"[{split_name}] n={stats['n']} positives={stats['positives']} prevalence={prevalence:.4f}")
        else:
            print(f"[{split_name}] n={stats['n']} positives={stats['positives']} prevalence=nan")

    model = build_model(
        args.input_mode,
        config=config,
        tabular_dim=len(config["tabular_features"]),
        lead_channels=(12 if args.input_mode == "full12" else len(lead_list) if lead_list else 1),
    ).to(device)
    pos_weight = compute_pos_weight(datasets["train"].label_array()).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    learning_rate = args.lr or config["training"]["lr"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config["training"]["weight_decay"],
    )

    output_root = Path(args.output_dir)
    models_dir = ensure_dir(output_root / "models")
    predictions_dir = ensure_dir(output_root / "predictions")
    run_name = make_run_name(
        args.label,
        args.input_mode,
        args.lead,
        args.seed,
        loader_info["train_fraction"],
        leads=lead_list,
    )
    run_model_dir = ensure_dir(models_dir / run_name)
    run_pred_dir = ensure_dir(predictions_dir / run_name)

    epochs = args.epochs or config["training"]["epochs"]
    patience = config["training"]["patience"]
    monitor_metric = config["training"]["monitor"]

    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, dataloaders["train"], optimizer, criterion, args.input_mode, device)

        val_predictions = predict(model, dataloaders["val"], args.input_mode, device, desc=f"Val epoch {epoch}")
        val_metrics = evaluate_predictions(val_predictions)
        val_logits = torch.tensor(val_predictions["logit"].to_numpy(), dtype=torch.float32)
        val_labels = torch.tensor(val_predictions["label"].to_numpy(), dtype=torch.float32)
        val_loss = float(F.binary_cross_entropy_with_logits(val_logits, val_labels, pos_weight=pos_weight.detach().cpu()).item())

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_brier_score": val_metrics["brier_score"],
        }
        history.append(epoch_log)

        monitor_value = val_loss if monitor_metric == "val_loss" else float(val_metrics["auroc"])
        if select_improved(monitor_metric, best_metric, monitor_value):
            best_metric = monitor_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improvement = 0
            val_predictions.to_csv(run_pred_dir / "val_predictions.csv", index=False)
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, run_model_dir / "best_model.pt")

    val_predictions = pd.read_csv(run_pred_dir / "val_predictions.csv")
    test_predictions = predict(model, dataloaders["test"], args.input_mode, device, desc="Test")
    test_predictions.to_csv(run_pred_dir / "test_predictions.csv", index=False)

    val_metrics = evaluate_predictions(val_predictions)
    test_metrics = evaluate_predictions(test_predictions)

    training_log = pd.DataFrame(history)
    training_log.to_csv(run_model_dir / "training_log.csv", index=False)

    metrics = {
        "label": args.label,
        "input_mode": args.input_mode,
        "lead": args.lead,
        "leads": lead_list,
        "seed": args.seed,
        "train_fraction": loader_info["train_fraction"],
        "learning_rate": learning_rate,
        "batch_size": loader_info["batch_size"],
        "best_epoch": best_epoch,
        "monitor": monitor_metric,
        "device": str(device),
        "splits": splits,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_dir": str(run_model_dir),
        "prediction_dir": str(run_pred_dir),
    }
    save_json(metrics, run_model_dir / "metrics.json")
    return metrics


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
