from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def make_run_name(label: str, input_mode: str, lead: str | None, seed: int, train_fraction: float) -> str:
    lead_part = lead if lead else "none"
    fraction_part = f"{train_fraction:.3f}".rstrip("0").rstrip(".")
    return f"{label}__{input_mode}__{lead_part}__frac_{fraction_part}__seed_{seed}"


def format_float(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "nan"
    return f"{value:.{ndigits}f}"
