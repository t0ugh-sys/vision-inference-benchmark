from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class InferenceBackend(Protocol):
    def warmup(self, tensor: np.ndarray, steps: int) -> None:
        ...

    def infer(self, tensor: np.ndarray) -> tuple[Any, float]:
        ...


class OnnxRuntimeBackend:
    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def warmup(self, tensor: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            self.session.run(None, {self.input_name: tensor})

    def infer(self, tensor: np.ndarray) -> tuple[Any, float]:
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return outputs, elapsed_ms


class UltralyticsBackend:
    def __init__(self, model_path: str) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)

    def warmup(self, tensor: np.ndarray, steps: int) -> None:
        dummy = np.transpose(tensor[0], (1, 2, 0))
        dummy = (dummy * 255.0).clip(0, 255).astype(np.uint8)
        for _ in range(steps):
            self.model.predict(dummy, verbose=False)

    def infer(self, tensor: np.ndarray) -> tuple[Any, float]:
        image = np.transpose(tensor[0], (1, 2, 0))
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
        start = time.perf_counter()
        results = self.model.predict(image, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return results, elapsed_ms


class TensorRTBackend:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        raise NotImplementedError("Fill TensorRT runtime binding in pipeline/backends.py on the target machine.")

    def warmup(self, tensor: np.ndarray, steps: int) -> None:
        raise NotImplementedError

    def infer(self, tensor: np.ndarray) -> tuple[Any, float]:
        raise NotImplementedError


class RKNNBackend:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        raise NotImplementedError("Fill RKNN runtime binding in pipeline/backends.py on the target machine.")

    def warmup(self, tensor: np.ndarray, steps: int) -> None:
        raise NotImplementedError

    def infer(self, tensor: np.ndarray) -> tuple[Any, float]:
        raise NotImplementedError


def build_backend(name: str, model_path: str) -> InferenceBackend:
    normalized = name.lower()
    if normalized == "pt":
        return UltralyticsBackend(model_path)
    if normalized == "onnx":
        return OnnxRuntimeBackend(model_path)
    if normalized == "tensorrt":
        return TensorRTBackend(model_path)
    if normalized == "rknn":
        return RKNNBackend(model_path)
    raise ValueError(f"Unsupported backend: {name}")
