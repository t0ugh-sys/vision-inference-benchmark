from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    tensor: np.ndarray
    meta: Dict[str, Any]
    elapsed_ms: float


def letterbox(image: np.ndarray, new_shape: int, stride: int = 32) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    shape = image.shape[:2]
    ratio = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return image, ratio, (left, top)


def preprocess_image(image: np.ndarray, config: Dict[str, Any]) -> PreprocessResult:
    start = time.perf_counter()
    cfg = config["preprocess"]
    imgsz = int(cfg["imgsz"])
    stride = int(cfg.get("stride", 32))

    processed, ratio, pad = letterbox(image, imgsz, stride)
    if cfg.get("bgr_to_rgb", True):
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    processed = processed.astype(np.float32)
    if cfg.get("normalize", True):
        mean = np.asarray(cfg.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        std = np.asarray(cfg.get("std", [255.0, 255.0, 255.0]), dtype=np.float32)
        processed = (processed - mean) / std

    tensor = np.transpose(processed, (2, 0, 1))[None, ...]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return PreprocessResult(
        tensor=tensor,
        meta={
            "ratio": ratio,
            "pad": pad,
            "original_shape": image.shape[:2],
            "input_shape": tensor.shape,
        },
        elapsed_ms=elapsed_ms,
    )
