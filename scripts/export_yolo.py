from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from pipeline.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO models to deployment formats.")
    parser.add_argument("--config", required=True, help="Path to pipeline yaml")
    parser.add_argument("--target", required=True, choices=["onnx", "tensorrt", "rknn"])
    parser.add_argument("--weights", help="Override pt weights path")
    parser.add_argument("--int8", action="store_true", help="Export quantized target where supported")
    parser.add_argument("--half", action="store_true", help="Export half precision target where supported")
    return parser.parse_args()


def export_with_ultralytics(config: dict, target: str, weights: str, half: bool, int8: bool) -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    export_cfg = config["export"]

    if target == "onnx":
        model.export(
            format="onnx",
            imgsz=int(config["model"]["imgsz"]),
            opset=int(export_cfg.get("opset", 12)),
            simplify=bool(export_cfg.get("simplify", True)),
            dynamic=bool(export_cfg.get("dynamic", False)),
            half=half,
            int8=int8,
        )
        return

    if target == "tensorrt":
        model.export(
            format="engine",
            imgsz=int(config["model"]["imgsz"]),
            workspace=float(export_cfg.get("workspace", 4)),
            half=half,
            int8=int8,
            data=config["dataset"]["data"],
        )
        return

    raise ValueError(f"Unsupported Ultralytics export target: {target}")


def export_rknn_stub(config: dict, weights: str, int8: bool) -> None:
    target = config["export"].get("rknn_target", "rk3588")
    print("RKNN export scaffold:")
    print(f"- source onnx/pt: {weights}")
    print(f"- target chip: {target}")
    print(f"- quantized: {int8}")
    print("- implement RKNN-Toolkit2 conversion in scripts/export_yolo.py on the RKNN machine")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    weights = args.weights or config["model"]["pt"]

    if not Path(weights).exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    if args.target in {"onnx", "tensorrt"}:
        export_with_ultralytics(config, args.target, weights, args.half, args.int8)
    else:
        export_rknn_stub(config, weights, args.int8)


if __name__ == "__main__":
    main()
