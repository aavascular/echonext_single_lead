from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_default_paths(project_root: str | Path | None = None) -> dict[str, Path]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "configs": root / "configs",
        "data_raw": root / "data" / "raw",
        "data_processed": root / "data" / "processed",
        "outputs": root / "outputs",
    }
