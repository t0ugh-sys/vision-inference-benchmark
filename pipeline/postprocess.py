from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class PostprocessResult:
    detections: List[Dict[str, Any]]
    elapsed_ms: float


def postprocess_predictions(raw_output: Any, meta: Dict[str, Any], config: Dict[str, Any]) -> PostprocessResult:
    start = time.perf_counter()
    detections: List[Dict[str, Any]] = []

    if isinstance(raw_output, np.ndarray):
        array = raw_output
    elif isinstance(raw_output, list) and raw_output:
        array = np.asarray(raw_output[0])
    else:
        array = np.asarray(raw_output)

    if array.ndim >= 2:
        limit = min(int(config["postprocess"].get("max_det", 300)), array.shape[-2])
        for idx in range(limit):
            row = array[0, idx] if array.ndim == 3 else array[idx]
            if row.shape[0] < 6:
                continue
            score = float(row[4])
            if score < float(config["postprocess"]["conf"]):
                continue
            detections.append(
                {
                    "bbox": [float(x) for x in row[:4]],
                    "score": score,
                    "class_id": int(row[5]),
                }
            )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PostprocessResult(detections=detections, elapsed_ms=elapsed_ms)
