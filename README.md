# vision-inference-benchmark

YOLO detection pipeline project for exporting, adapting, benchmarking, and comparing `pt`, `onnx`, `TensorRT`, and `RKNN` models.

## Goals

- Keep one config for preprocessing, graph adaptation, inference, postprocessing, and reporting
- Export YOLO models to `onnx`, `engine`, and `rknn` workflows
- Add ONNX graph rewrite and operator adaptation before deployment conversion
- Measure accuracy and latency before and after quantization
- Report preprocessing time, model inference time, postprocessing time, and total time
- Compare metric drops after conversion and quantization

## Layout

```text
vision-inference-benchmark/
|- README.md
|- requirements.txt
|- configs/
|  |- pipeline.yaml
|- pipeline/
|  |- __init__.py
|  |- backends.py
|  |- benchmark.py
|  |- config.py
|  |- graph_rewrite.py
|  |- metrics.py
|  |- postprocess.py
|  |- preprocess.py
|  |- report.py
|- reports/
|- scripts/
|  |- adapt_model.py
|  |- benchmark_model.py
|  |- check_backend_support.py
|  |- compare_reports.py
|  |- export_yolo.py
|  |- run_cpp_benchmark_matrix.py
|  |- summarize_benchmarks.py
|- cpp/
```

## Quick start

1. Update `configs/pipeline.yaml`.
2. Export and adapt ONNX for TensorRT:

```bash
python scripts/export_yolo.py --config configs/pipeline.yaml --target tensorrt --adapt --half
```

3. Export and adapt ONNX for RKNN:

```bash
python scripts/export_yolo.py --config configs/pipeline.yaml --target rknn --adapt --int8
```

4. Audit backend operator support before deployment:

```bash
python scripts/check_backend_support.py --input weights/test_model.onnx --backend tensorrt --report reports/test_model.support.json
```

5. Run benchmark:

```bash
python scripts/benchmark_model.py --config configs/pipeline.yaml --backend onnx --model weights/adapted/yolo11n.tensorrt.onnx
python scripts/benchmark_model.py --config configs/pipeline.yaml --backend tensorrt --model weights/yolo11n.engine
```

6. Summarize reports:

```bash
python scripts/summarize_benchmarks.py --reports-dir reports --output-prefix reports/summary
```

## Operator Adaptation

For deployment benchmarking, operator adaptation should happen after ONNX export and before TensorRT or RKNN conversion.

The current adaptation stage supports:

- ONNX shape inference
- Identity node stripping
- Optional opset conversion
- Optional `onnxsim` simplify
- Custom activation rewrite for `SiLU`, `Swish`, and `HardSwish`
- Backend-specific `Resize` normalization
- Backend compatibility report for TensorRT and RKNN sensitive ops
- Custom-op detection report
- Optional `fail_on_blocked` behavior for CI or deployment gating

Run it directly with:

```bash
python scripts/adapt_model.py --config configs/pipeline.yaml --input weights/yolo11n.onnx --backend tensorrt
python scripts/adapt_model.py --config configs/pipeline.yaml --input weights/yolo11n.onnx --backend rknn --output weights/adapted/yolo11n.rknn.onnx
```

Outputs:

- Adapted ONNX model under `weights/adapted/`
- JSON rewrite report under `reports/`

The report now includes:

- `status`
- `stage`
- `error`
- `rewrites_applied`
- `op_histogram`
- `compatibility.sensitive_ops`
- `compatibility.custom_ops`
- `compatibility.blocked`
- `artifacts`

## Support Audit

`check_backend_support.py` audits a model without rewriting it.

Examples:

```bash
python scripts/check_backend_support.py --input weights/test_model.onnx --backend tensorrt
python scripts/check_backend_support.py --input weights/test_model.onnx --backend rknn --report reports/test_model.rknn.support.json --fail-on-blocked
```

Use this when you want a fast operator-compatibility verdict before export, adaptation, or deployment.

## Export -> Adapt -> Convert

`scripts/export_yolo.py` now supports chaining these stages.

Examples:

```bash
python scripts/export_yolo.py --config configs/pipeline.yaml --target onnx --adapt
python scripts/export_yolo.py --config configs/pipeline.yaml --target tensorrt --adapt --half
python scripts/export_yolo.py --config configs/pipeline.yaml --target rknn --adapt --int8
```

Behavior:

- `target=onnx`: export ONNX, then optionally adapt it for ONNX Runtime
- `target=tensorrt`: if `--adapt` is enabled, export ONNX, adapt for TensorRT, then build an engine with `trtexec`
- `target=rknn`: export ONNX, adapt for RKNN, then print the adapted ONNX path for RKNN-Toolkit2 conversion

TensorRT builder settings are configured in `configs/pipeline.yaml` under `deploy.tensorrt`.

## C++ benchmark pipeline

The project also includes a C++ benchmark scaffold in `cpp/` for measuring inference speed across exported model formats.

### Build

Dependencies:

- CMake 3.20+
- OpenCV
- yaml-cpp
- ONNX Runtime
- TensorRT if you want `.engine` benchmarking

Example build for ONNX Runtime:

```bash
cd cpp
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime -DPIPELINE_ENABLE_ORT=ON -DPIPELINE_ENABLE_TENSORRT=OFF
cmake --build build --config Release
```

Example build for ONNX Runtime + TensorRT:

```bash
cd cpp
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime -DTENSORRT_ROOT=/path/to/tensorrt -DPIPELINE_ENABLE_ORT=ON -DPIPELINE_ENABLE_TENSORRT=ON
cmake --build build --config Release
```

### Batch benchmarking

```bash
python scripts/run_cpp_benchmark_matrix.py --exe cpp/build/Release/pipeline_benchmark.exe --config configs/pipeline.yaml --onnx-fp16 weights/yolo11n_fp16.onnx --onnx-int8 weights/yolo11n_int8.onnx --trt-fp16 weights/yolo11n_fp16.engine --trt-int8 weights/yolo11n_int8.engine --summary-prefix reports/cpp_matrix_summary
```

## Notes

- `onnxruntime` benchmarking uses the available CUDA provider first, then CPU fallback.
- TensorRT and RKNN still need target-machine runtime validation.
- `onnxsim` is optional and only required when you enable simplify in the adaptation stage.
- If `trtexec` is not in `PATH`, set `deploy.tensorrt.builder` to the full executable path.
