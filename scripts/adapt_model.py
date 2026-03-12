from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite and adapt an ONNX model for a target backend.")
    parser.add_argument("--config", required=True, help="Path to pipeline yaml")
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--backend", required=True, choices=["onnxruntime", "tensorrt", "rknn"])
    parser.add_argument("--output", help="Output ONNX model path")
    parser.add_argument("--report", help="Output JSON report path")
    parser.add_argument("--opset", type=int, help="Override target opset")
    parser.add_argument("--skip-infer-shapes", action="store_true", help="Disable ONNX shape inference")
    parser.add_argument("--skip-strip-identity", action="store_true", help="Disable Identity node removal")
    parser.add_argument("--simplify", action="store_true", help="Enable onnxsim simplify pass")
    parser.add_argument("--skip-compatibility-check", action="store_true", help="Disable backend compatibility warnings")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Fail when compatibility report is blocked")
    return parser.parse_args()


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def main() -> None:
    args = parse_args()

    from pipeline.config import load_config
    from pipeline.graph_rewrite import rewrite_onnx_model

    config_path = resolve_path(ROOT, args.config)
    config = load_config(config_path)
    adapt_cfg = config.get("adapt", {})
    project_root = config_path.parent.parent

    input_path = resolve_path(project_root, args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    default_output_dir = resolve_path(project_root, adapt_cfg.get("output_dir", "weights/adapted"))
    output_path = resolve_path(project_root, args.output) if args.output else (default_output_dir / f"{input_path.stem}.{args.backend}.onnx").resolve()

    report_dir = resolve_path(project_root, config["project"].get("report_dir", "reports"))
    report_path = resolve_path(project_root, args.report) if args.report else (report_dir / f"{output_path.stem}.rewrite.json").resolve()

    target_opset = args.opset if args.opset is not None else adapt_cfg.get("target_opset")
    if isinstance(target_opset, str):
        target_opset = int(target_opset)

    result = rewrite_onnx_model(
        input_path=input_path,
        output_path=output_path,
        backend=args.backend,
        target_opset=target_opset,
        infer_shapes_first=not args.skip_infer_shapes and bool(adapt_cfg.get("infer_shapes", True)),
        strip_identity=not args.skip_strip_identity and bool(adapt_cfg.get("strip_identity", True)),
        simplify_graph=bool(args.simplify or adapt_cfg.get("simplify", False)),
        simplify_check_n=int(adapt_cfg.get("simplify_check_n", 0)),
        compatibility_check=not args.skip_compatibility_check and bool(adapt_cfg.get("compatibility_check", True)),
        fail_on_blocked=args.fail_on_blocked or bool(adapt_cfg.get("fail_on_blocked", False)),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Adapted model saved to {output_path}")
    print(f"Rewrite report saved to {report_path}")
    if result.warnings:
        print("Warnings:")
        for item in result.warnings:
            print(f"- {item}")


if __name__ == "__main__":
    main()
