from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def resolve_path(base_dir: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_project_config(path: str | Path) -> tuple[Path, Dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = load_config(config_path)
    project_root = config_path.parent.parent
    return config_path, config, project_root


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_report_dir(config: Dict[str, Any]) -> Path:
    return ensure_directory(config["project"]["report_dir"])
