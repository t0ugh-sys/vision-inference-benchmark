# vision-inference-benchmark

YOLO detection pipeline project for exporting, adapting, benchmarking, and comparing `pt`, `onnx`, `TensorRT`, and `RKNN` models.

## Goals

- Keep one config for preprocessing, graph adaptation, inference, postprocessing, and reporting
- Export YOLO models to `onnx`, `engine`, and `rknn` workflows
- Add ONNX graph rewrite / operator adaptation before deployment conversion
- Measure accuracy and latency before and after quantization
- Report preprocessing time, model inference time, postprocessing time, and total time
- Compare metric drops after conversion and quantization

## Layout

```text
vision-inference-benchmark/
©À©¤ README.md
©À©¤ requirements.txt
©À©¤ configs/
©¦  ©¸©¤ pipeline.yaml
©À©¤ pipeline/
©¦  ©À©¤ __init__.py
©¦  ©À©¤ backends.py
©¦  ©À©¤ benchmark.py
©¦  ©À©¤ config.py
©¦  ©À©¤ graph_rewrite.py
©¦  ©À©¤ metrics.py
©¦  ©À©¤ postprocess.py
©¦  ©À©¤ preprocess.py
©¦  ©¸©¤ report.py
©À©¤ reports/
©À©¤ scripts/
©¦  ©À©¤ adapt_model.py
©¦  ©À©¤ benchmark_model.py
©¦  ©À©¤ compare_reports.py
©¦  ©À©¤ export_yolo.py
©¦  ©À©¤ run_cpp_benchmark_matrix.py
©¦  ©¸©¤ summarize_benchmarks.py
©¸©¤ cpp/
```

## What this project tracks

- Accuracy:
  - `mAP50`
  - `mAP50_95`
  - `precision`
  - `recall`
- Latency:
  - `preprocess_ms`
  - `inference_ms`
  - `postprocess_ms`
  - `total_ms`
  - `fps`
- Conversion:
  - PyTorch FP32 baseline
  - ONNX FP32 / INT8
  - TensorRT FP16 / INT8
  - RKNN FP16 / INT8
- Degradation:
  - metric drop after export
  - metric drop after quantization

## Quick start

1. Update `configs/pipeline.yaml`
2. Export models:

```bash
python scripts/export_yolo.py --config configs/pipeline.yaml --target onnx
python scripts/export_yolo.py --config configs/pipeline.yaml --target tensorrt
python scripts/export_yolo.py --config configs/pipeline.yaml --target rknn
```

3. Adapt ONNX operators for target backend:

```bash
python scripts/adapt_model.py --config configs/pipeline.yaml --input weights/yolo11n.onnx --backend tensorrt
python scripts/adapt_model.py --config configs/pipeline.yaml --input weights/yolo11n.onnx --backend rknn
```

4. Run benchmark:

```bash
python scripts/benchmark_model.py --config configs/pipeline.yaml --backend pt --model weights/yolo11n.pt
python scripts/benchmark_model.py --config configs/pipeline.yaml --backend onnx --model weights/yolo11n.onnx
```

5. Compare reports:

```bash
python scripts/compare_reports.py --baseline reports/pt_fp32.json --candidate reports/tensorrt_int8.json
```

6. Summarize multiple reports:

```bash
python scripts/summarize_benchmarks.py --reports-dir reports --output-prefix reports/summary
python scripts/summarize_benchmarks.py --reports-dir reports --baseline reports/onnx_fp16_cpp.json --output-prefix reports/summary_vs_onnx_fp16
```

7. Run a full C++ benchmark matrix and summarize automatically:

```bash
python scripts/run_cpp_benchmark_matrix.py ^
  --exe cpp/build/Release/pipeline_benchmark.exe ^
  --config configs/pipeline.yaml ^
  --onnx-fp16 weights/yolo11n_fp16.onnx ^
  --onnx-int8 weights/yolo11n_int8.onnx ^
  --trt-fp16 weights/yolo11n_fp16.engine ^
  --trt-int8 weights/yolo11n_int8.engine ^
  --summary-prefix reports/cpp_matrix_summary
```

## Operator Adaptation

For deployment-focused benchmarking, operator adaptation should happen after ONNX export and before TensorRT or RKNN conversion.

The current adaptation stage supports:

- ONNX shape inference
- Identity node stripping
- Optional opset conversion
- Optional `onnxsim` simplify
- Backend compatibility warnings for TensorRT / RKNN sensitive ops

Run it with:

```bash
python scripts/adapt_model.py --config configs/pipeline.yaml --input weights/yolo11n.onnx --backend tensorrt
python scripts/adapt_model.py --config configs/pipeline.yaml --input weights/yolo11n.onnx --backend rknn --output weights/adapted/yolo11n.rknn.onnx
```

Outputs:

- adapted ONNX model
- JSON rewrite report under `reports/`

This stage is implemented in `pipeline/graph_rewrite.py` and is the right place to add future model-specific rules such as custom op replacement, plugin fallback tagging, or backend-specific graph normalization.

## C++ benchmark pipeline

The project also includes a C++ benchmark scaffold in `cpp/` for measuring inference speed across exported model formats.

### What the C++ version covers

- Shared preprocessing / postprocessing / JSON reporting
- ONNX Runtime backend for `.onnx`
- TensorRT backend for `.engine`
- Precision tagging for `fp16` and `int8`
- Same timing fields as the Python pipeline:
  - `preprocess_ms`
  - `inference_ms`
  - `postprocess_ms`
  - `total_ms`
  - `fps`

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

### Run

ONNX FP16:

```bash
cpp/build/pipeline_benchmark --config configs/pipeline.yaml --backend onnx --model weights/yolo11n_fp16.onnx --precision fp16 --report reports/onnx_fp16_cpp.json
```

ONNX INT8:

```bash
cpp/build/pipeline_benchmark --config configs/pipeline.yaml --backend onnx --model weights/yolo11n_int8.onnx --precision int8 --quantized --report reports/onnx_int8_cpp.json
```

TensorRT FP16:

```bash
cpp/build/pipeline_benchmark --config configs/pipeline.yaml --backend tensorrt --model weights/yolo11n_fp16.engine --precision fp16 --report reports/tensorrt_fp16_cpp.json
```

TensorRT INT8:

```bash
cpp/build/pipeline_benchmark --config configs/pipeline.yaml --backend tensorrt --model weights/yolo11n_int8.engine --precision int8 --quantized --report reports/tensorrt_int8_cpp.json
```

Batch summary after running several benchmarks:

```bash
python scripts/summarize_benchmarks.py --reports-dir reports --output-prefix reports/cpp_summary
python scripts/summarize_benchmarks.py --reports-dir reports --baseline reports/tensorrt_fp16_cpp.json --output-prefix reports/cpp_summary_vs_trt_fp16
```

One-shot matrix run:

```bash
python scripts/run_cpp_benchmark_matrix.py --exe cpp/build/Release/pipeline_benchmark.exe --config configs/pipeline.yaml --onnx-fp16 weights/yolo11n_fp16.onnx --onnx-int8 weights/yolo11n_int8.onnx --trt-fp16 weights/yolo11n_fp16.engine --trt-int8 weights/yolo11n_int8.engine --summary-prefix reports/cpp_matrix_summary
```

### Notes

- Precision is determined by the exported model or engine you pass in; the `--precision` argument is recorded into the report for comparison.
- ONNX Runtime inference here is focused on speed benchmarking. If you need CUDA EP or TensorRT EP, extend the backend initialization in `cpp/src/main.cpp`.
- TensorRT code assumes a single input and a single output binding, which matches many exported YOLO deployment graphs. If your engine has multiple outputs, extend the binding handling before production use.
- On Windows multi-config generators, the executable is usually under `cpp/build/Release/pipeline_benchmark.exe`.
- `scripts/summarize_benchmarks.py` generates `.json`, `.csv`, and `.md` summary files so you can directly compare `onnx/tensorrt` and `fp16/int8`.
- `scripts/run_cpp_benchmark_matrix.py` runs multiple C++ benchmark cases in sequence and then generates a consolidated summary automatically.

## Notes

- `onnx` benchmarking uses `onnxruntime` if available.
- `TensorRT` and `RKNN` runners are scaffolded with clear hook points; backend package integration can be filled in on your target machine.
- Accuracy evaluation is expected to come from your validation output or framework validator, then merged into the benchmark report.
