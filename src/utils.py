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


def make_lead_descriptor(lead: str | None = None, leads: list[str] | None = None) -> str:
    if leads:
        return "-".join(leads)
    if lead:
        return lead
    return "none"


def parse_leads(leads_arg: str | None) -> list[str] | None:
    if leads_arg is None:
        return None
    parsed = [lead.strip() for lead in leads_arg.split(",") if lead.strip()]
    return parsed or None


def make_run_name(
    label: str,
    input_mode: str,
    lead: str | None,
    seed: int,
    train_fraction: float,
    leads: list[str] | None = None,
) -> str:
    lead_part = make_lead_descriptor(lead=lead, leads=leads)
    fraction_part = f"{train_fraction:.3f}".rstrip("0").rstrip(".")
    return f"{label}__{input_mode}__{lead_part}__frac_{fraction_part}__seed_{seed}"


def format_float(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "nan"
    return f"{value:.{ndigits}f}"
