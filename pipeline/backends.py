from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class InferenceBackend(Protocol):
    def warmup(self, tensor: np.ndarray, steps: int) -> None:
        ...

    def infer(self, tensor: np.ndarray) -> tuple[Any, float]:
        ...


class OnnxRuntimeBackend:
    def __init__(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        providers = ort.get_available_providers()
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
        logger.info(f"Using ONNX Runtime provider: {provider}")

        self.session = ort.InferenceSession(model_path, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"ONNX model loaded: {self.input_name} shape={self.session.get_inputs()[0].shape}")

    def warmup(self, tensor: np.ndarray, steps: int) -> None:
        logger.debug(f"Warming up ONNX runtime with {steps} iterations")
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
    def __init__(self, model_path: str, enable_int8: bool = False) -> None:
        import tensorrt as trt

        self.model_path = Path(model_path)
        self.enable_int8 = enable_int8

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        # Initialize pycuda
        import pycuda.driver as cuda
        cuda.init()
        self._cuda = cuda
        self._stream = cuda.Stream()

        # Allocate buffers
        self.inputs: list[Any] = []
        self.outputs: list[Any] = []
        self.bindings: list[int] = []

        for i in range(self.engine.num_bindings):
            binding = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape) * dtype.itemsize
            if self.engine.binding_is_input(binding):
                self.inputs.append(cuda.mem_alloc(size))
            else:
                self.outputs.append(cuda.mem_alloc(size))
            self.bindings.append(int(self.inputs[-1]) if self.engine.binding_is_input(binding) else int(self.outputs[-1]))

        # Pre-allocate host output buffer
        self._output_shape = self.engine.get_binding_shape(self.engine.get_binding_name(1))
        self._output_dtype = trt.nptype(self.engine.get_binding_dtype(self.engine.get_binding_name(1)))

    def warmup(self, tensor: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            self._inference_internal(tensor)

    def infer(self, tensor: np.ndarray) -> tuple[Any, float]:
        start = time.perf_counter()
        outputs = self._inference_internal(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return outputs, elapsed_ms

    def _inference_internal(self, tensor: np.ndarray) -> list[np.ndarray]:
        cuda = self._cuda
        stream = self._stream

        # Ensure tensor is contiguous and correct dtype
        tensor = np.ascontiguousarray(tensor)

        # Copy input to GPU
        cuda.memcpy_htod_async(self.inputs[0], tensor.ravel(), stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=stream.handle)

        # Copy output from GPU
        host_output = np.empty(self._output_shape, dtype=self._output_dtype)
        cuda.memcpy_dtoh_async(host_output, self.outputs[0], stream)
        stream.synchronize()

        return [host_output]


class RKNNBackend:
    def __init__(self, model_path: str, target_device: str = "rv1106") -> None:
        try:
            import rknn.api as rknn
        except ImportError:
            raise ImportError("RKNN toolkit not installed. Install with: pip install rknn-toolkit2")

        self.model_path = Path(model_path)
        self._rknn = rknn

        self.rknn = rknn.RKNN()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")

        ret = self.rknn.init_runtime(target=target_device)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize RKNN runtime: {ret}")

    def warmup(self, tensor: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            self.rknn.inference(inputs=[tensor])

    def infer(self, tensor: np.ndarray) -> tuple[Any, float]:
        tensor = np.ascontiguousarray(tensor)
        start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[tensor])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return outputs, elapsed_ms

    def __del__(self):
        if hasattr(self, "rknn"):
            self.rknn.release()


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
