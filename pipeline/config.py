from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_report_dir(config: Dict[str, Any]) -> Path:
    report_dir = Path(config["project"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir
