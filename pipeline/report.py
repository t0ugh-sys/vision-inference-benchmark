from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pipeline.config import ensure_directory


def save_report(path: str | Path, data: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def load_report(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
