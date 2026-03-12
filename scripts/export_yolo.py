from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO models to deployment formats.")
    parser.add_argument("--config", required=True, help="Path to pipeline yaml")
    parser.add_argument("--target", required=True, choices=["onnx", "tensorrt", "rknn"])
    parser.add_argument("--weights", help="Override pt weights path")
    parser.add_argument("--int8", action="store_true", help="Export quantized target where supported")
    parser.add_argument("--half", action="store_true", help="Export half precision target where supported")
    parser.add_argument("--adapt", action="store_true", help="Run ONNX adaptation after export")
    parser.add_argument("--skip-adapt", action="store_true", help="Disable ONNX adaptation even if enabled in config")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Fail when adaptation marks the graph as blocked")
    return parser.parse_args()


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def export_onnx_with_ultralytics(config: dict, weights: str, half: bool, int8: bool) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    export_cfg = config["export"]
    output = model.export(
        format="onnx",
        imgsz=int(config["model"]["imgsz"]),
        opset=int(export_cfg.get("opset", 12)),
        simplify=bool(export_cfg.get("simplify", True)),
        dynamic=bool(export_cfg.get("dynamic", False)),
        half=half,
        int8=int8,
    )
    return Path(output).resolve()


def export_engine_with_ultralytics(config: dict, weights: str, half: bool, int8: bool) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    export_cfg = config["export"]
    output = model.export(
        format="engine",
        imgsz=int(config["model"]["imgsz"]),
        workspace=float(export_cfg.get("workspace", 4)),
        half=half,
        int8=int8,
        data=config["dataset"]["data"],
    )
    return Path(output).resolve()


def adapt_onnx(config: dict, project_root: Path, input_path: Path, backend: str, fail_on_blocked: bool) -> tuple[Path, Path, dict]:
    from pipeline.graph_rewrite import rewrite_onnx_model

    adapt_cfg = config.get("adapt", {})
    output_dir = resolve_path(project_root, adapt_cfg.get("output_dir", "weights/adapted"))
    output_path = (output_dir / f"{input_path.stem}.{backend}.onnx").resolve()
    report_dir = resolve_path(project_root, config["project"].get("report_dir", "reports"))
    report_path = (report_dir / f"{output_path.stem}.rewrite.json").resolve()

    result = rewrite_onnx_model(
        input_path=input_path,
        output_path=output_path,
        backend=backend,
        target_opset=adapt_cfg.get("target_opset"),
        infer_shapes_first=bool(adapt_cfg.get("infer_shapes", True)),
        strip_identity=bool(adapt_cfg.get("strip_identity", True)),
        simplify_graph=bool(adapt_cfg.get("simplify", False)),
        simplify_check_n=int(adapt_cfg.get("simplify_check_n", 0)),
        compatibility_check=bool(adapt_cfg.get("compatibility_check", True)),
        rewrite_custom_ops=bool(adapt_cfg.get("rewrite_custom_ops", True)),
        normalize_resize=bool(adapt_cfg.get("normalize_resize", True)),
        fail_on_blocked=fail_on_blocked or bool(adapt_cfg.get("fail_on_blocked", False)),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Adapted ONNX saved to {output_path}")
    print(f"Adaptation report saved to {report_path}")
    return output_path, report_path, result.to_dict()


def resolve_tensorrt_builder(config: dict) -> Path:
    deploy_cfg = config.get("deploy", {}).get("tensorrt", {})
    builder = deploy_cfg.get("builder", "trtexec")
    resolved = shutil.which(builder)
    if resolved:
        return Path(resolved).resolve()

    builder_path = Path(str(builder))
    if builder_path.exists():
        return builder_path.resolve()

    raise FileNotFoundError("TensorRT builder not found. Configure deploy.tensorrt.builder or install trtexec.")


def build_tensorrt_engine(config: dict, project_root: Path, onnx_path: Path, half: bool, int8: bool) -> Path:
    deploy_cfg = config.get("deploy", {}).get("tensorrt", {})
    builder_path = resolve_tensorrt_builder(config)

    default_engine = config["model"].get("tensorrt", onnx_path.with_suffix(".engine").as_posix())
    engine_path = resolve_path(project_root, deploy_cfg.get("engine_path", default_engine))
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    workspace_gb = float(config["export"].get("workspace", 4))
    workspace_mb = int(workspace_gb * 1024)
    command = [
        str(builder_path),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
    ]
    if half:
        command.append("--fp16")
    if int8:
        command.append("--int8")

    extra_args = deploy_cfg.get("extra_args", [])
    if isinstance(extra_args, list):
        command.extend(str(item) for item in extra_args)

    print("Running TensorRT builder:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    return engine_path


def export_rknn_stub(config: dict, source_onnx: Path, int8: bool) -> None:
    target = config["export"].get("rknn_target", "rk3588")
    print("RKNN export scaffold:")
    print(f"- source onnx: {source_onnx}")
    print(f"- target chip: {target}")
    print(f"- quantized: {int8}")
    print("- next step: convert this adapted ONNX with RKNN-Toolkit2 on the RKNN machine")


def main() -> None:
    args = parse_args()

    from pipeline.config import load_config

    config_path = resolve_path(ROOT, args.config)
    config = load_config(config_path)
    project_root = config_path.parent.parent

    weights = resolve_path(project_root, args.weights or config["model"]["pt"])
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    adapt_enabled = (args.adapt or bool(config.get("adapt", {}).get("enabled", False))) and not args.skip_adapt

    if args.target == "onnx":
        onnx_path = export_onnx_with_ultralytics(config, str(weights), args.half, args.int8)
        print(f"Exported ONNX model to {onnx_path}")
        if adapt_enabled:
            adapt_onnx(config, project_root, onnx_path, "onnxruntime", args.fail_on_blocked)
        return

    if args.target == "tensorrt":
        if adapt_enabled:
            onnx_path = export_onnx_with_ultralytics(config, str(weights), args.half, args.int8)
            print(f"Exported ONNX model to {onnx_path}")
            adapted_onnx, _, _ = adapt_onnx(config, project_root, onnx_path, "tensorrt", args.fail_on_blocked)
            engine_path = build_tensorrt_engine(config, project_root, adapted_onnx, args.half, args.int8)
            print(f"Built TensorRT engine at {engine_path}")
        else:
            resolve_tensorrt_builder(config)
            engine_path = export_engine_with_ultralytics(config, str(weights), args.half, args.int8)
            print(f"Exported TensorRT engine to {engine_path}")
        return

    onnx_path = export_onnx_with_ultralytics(config, str(weights), args.half, args.int8)
    print(f"Exported ONNX model to {onnx_path}")
    source_for_rknn = onnx_path
    if adapt_enabled:
        source_for_rknn, _, _ = adapt_onnx(config, project_root, onnx_path, "rknn", args.fail_on_blocked)
    export_rknn_stub(config, source_for_rknn, args.int8)


if __name__ == "__main__":
    main()
