from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a C++ benchmark matrix and generate summary files.")
    parser.add_argument("--exe", required=True, help="Path to pipeline_benchmark executable.")
    parser.add_argument("--config", required=True, help="Path to pipeline yaml.")
    parser.add_argument("--reports-dir", default="reports", help="Directory for generated reports.")
    parser.add_argument("--images", help="Optional validation images directory override.")
    parser.add_argument("--warmup", type=int, help="Optional warmup override.")
    parser.add_argument("--runs", type=int, help="Optional run count override.")
    parser.add_argument("--baseline", help="Optional baseline report used for summary deltas.")
    parser.add_argument("--summary-prefix", default="reports/cpp_matrix_summary", help="Summary output prefix.")

    parser.add_argument("--onnx-fp16", help="Path to ONNX FP16 model.")
    parser.add_argument("--onnx-int8", help="Path to ONNX INT8 model.")
    parser.add_argument("--trt-fp16", help="Path to TensorRT FP16 engine.")
    parser.add_argument("--trt-int8", help="Path to TensorRT INT8 engine.")
    return parser.parse_args()


def resolve_path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def build_cases(args: argparse.Namespace, root: Path, reports_dir: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []

    def add_case(name: str, backend: str, precision: str, quantized: bool, model_arg: str | None) -> None:
        model_path = resolve_path(model_arg, root)
        if model_path is None:
            return
        cases.append(
            {
                "name": name,
                "backend": backend,
                "precision": precision,
                "quantized": quantized,
                "model_path": model_path,
                "report_path": reports_dir / f"{name}.json",
            }
        )

    add_case("onnx_fp16_cpp", "onnx", "fp16", False, args.onnx_fp16)
    add_case("onnx_int8_cpp", "onnx", "int8", True, args.onnx_int8)
    add_case("tensorrt_fp16_cpp", "tensorrt", "fp16", False, args.trt_fp16)
    add_case("tensorrt_int8_cpp", "tensorrt", "int8", True, args.trt_int8)
    return cases


def run_case(exe: Path, config: Path, images: Path | None, warmup: int | None, runs: int | None, case: dict[str, object]) -> None:
    command = [
        str(exe),
        "--config",
        str(config),
        "--backend",
        str(case["backend"]),
        "--model",
        str(case["model_path"]),
        "--precision",
        str(case["precision"]),
        "--report",
        str(case["report_path"]),
    ]

    if bool(case["quantized"]):
        command.append("--quantized")
    if images is not None:
        command.extend(["--images", str(images)])
    if warmup is not None:
        command.extend(["--warmup", str(warmup)])
    if runs is not None:
        command.extend(["--runs", str(runs)])

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def run_summary(reports_dir: Path, summary_prefix: Path, baseline: Path | None) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_benchmarks.py"),
        "--reports-dir",
        str(reports_dir),
        "--pattern",
        "*_cpp.json",
        "--output-prefix",
        str(summary_prefix),
    ]
    if baseline is not None:
        command.extend(["--baseline", str(baseline)])

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = ROOT

    exe = resolve_path(args.exe, root)
    config = resolve_path(args.config, root)
    reports_dir = resolve_path(args.reports_dir, root)
    images = resolve_path(args.images, root)
    summary_prefix = resolve_path(args.summary_prefix, root)
    baseline = resolve_path(args.baseline, root)

    if exe is None or not exe.exists():
        raise FileNotFoundError(f"Benchmark executable not found: {args.exe}")
    if config is None or not config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if reports_dir is None:
        raise FileNotFoundError("Reports directory could not be resolved")

    reports_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args, root, reports_dir)
    if not cases:
        raise ValueError("No benchmark cases configured. Pass at least one of --onnx-fp16, --onnx-int8, --trt-fp16, --trt-int8.")

    for case in cases:
        run_case(exe, config, images, args.warmup, args.runs, case)

    if baseline is None:
        baseline = Path(cases[0]["report_path"])

    run_summary(reports_dir, summary_prefix, baseline)


if __name__ == "__main__":
    main()
