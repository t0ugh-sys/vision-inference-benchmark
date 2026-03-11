from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.benchmark import run_benchmark, run_ultralytics_accuracy
from pipeline.config import ensure_report_dir, load_config
from pipeline.report import save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a YOLO model backend.")
    parser.add_argument("--config", required=True, help="Path to pipeline yaml")
    parser.add_argument("--backend", required=True, choices=["pt", "onnx", "tensorrt", "rknn"])
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--images", help="Validation images directory override")
    parser.add_argument("--report-name", help="Output report filename")
    parser.add_argument("--quantized", action="store_true", help="Mark report as quantized")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    image_dir = args.images or config["dataset"]["val_images"]
    report = run_benchmark(config, args.backend, args.model, image_dir)

    if config.get("accuracy", {}).get("enabled", False):
        try:
            accuracy = run_ultralytics_accuracy(config, args.model)
            report["accuracy"] = accuracy.to_dict()
        except Exception as exc:
            report["accuracy_error"] = str(exc)

    report["quantized"] = bool(args.quantized)

    report_dir = ensure_report_dir(config)
    report_name = args.report_name or config["benchmark"].get("report_name", f"{args.backend}.json")
    save_report(report_dir / Path(report_name).name, report)
    print(f"Saved report to {report_dir / Path(report_name).name}")


if __name__ == "__main__":
    main()
